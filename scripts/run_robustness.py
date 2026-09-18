#!/usr/bin/env python3
"""External-validity experiments that accompany the frozen study.

    asynchronous   lockstep against asynchronous progress (within groups, and fully dynamic)
    budget         synthesis-chain length (states per rotation) from 5 to 200
    staged         stage-resolved service laws with early rejection
    external       64 independently produced demand traces (BARC artifact of Ye, Khan and Liang)

Each part can be run on its own (--part NAME); --part merge combines the pieces into
data/results/robustness.json and data/results/raw_robustness_runs.csv.gz. The default runs
everything and merges. Total run time is about ten minutes on one core.
"""
from __future__ import annotations
import argparse, csv, glob, gzip, json, math, os, time
from pathlib import Path
import numpy as np
from traceq.model import Config, simulate, sample_latencies
from traceq.analysis import summarize
from traceq.asynchronous import rotation_graph, simulate_lanes
from traceq.workloads import hubbard_rotations, schedule
from traceq.protocols import Stage, staged_moments, sample_staged_fast
from traceq.theory import deterministic_runtime, fluid_runtime, histogram_envelope
from traceq.external import load_trace

ROOT = Path(__file__).resolve().parents[1]
TMP = ROOT / 'data/results/_robustness_parts'
PARTS = ['asynchronous', 'budget', 'staged', 'external']
TAU, ATTEMPT, P0, MU = 17.0, 9.0, 0.08, 112.5


def stats(values, ref):
    s = summarize(values)
    return dict(n=s['n'], excess=100 * (s['mean'] / ref - 1), lo=100 * (s['mean_ci_low'] / ref - 1),
                hi=100 * (s['mean_ci_high'] / ref - 1), cv=100 * s['cv'])


def workloads():
    work = {}
    for mp in ('JW', 'TT'):
        terms = hubbard_rotations(6, 4, mp)
        demand, groups = schedule(terms)
        work[mp] = dict(demand=demand, groups=groups, graph=rotation_graph(terms),
                        sizes=np.array([len(g) for g in groups], dtype=np.int64))
    return work


def part_asynchronous(runs, wr):
    work, rows = workloads(), []
    for mi, mp in enumerate(('JW', 'TT')):
        w = work[mp]; m = w['demand']; nT, S = int(m.sum()), len(m); t0 = S * TAU
        kavg = math.ceil(nT * MU / t0)
        for ki, K in enumerate((8, kavg, kavg + 2, kavg + 4, kavg + 8)):
            trate = max(t0, nT * MU / K); n = runs if K == kavg else runs // 2
            base = 13000000 + 100000 * mi + 10000 * ki
            vals = dict(lockstep=[], barrier=[], dynamic=[]); cfg = Config(K, 4)
            for i in range(n):
                sv = sample_latencies(cfg, nT + 300, base + i)       # common random numbers
                vals['lockstep'].append(simulate(m, cfg, services=sv)['makespan'])
                for mode in ('barrier', 'dynamic'):
                    vals[mode].append(simulate_lanes(w['graph'], w['groups'], K, 4, TAU, sv, mode)['makespan'])
                for mode in vals:
                    wr.writerow([f'async_{mp}_{K}_{mode}', i, base + i, vals[mode][-1]])
            svd = np.full((K, nT + 300), MU)
            det = dict(lockstep=simulate(m, Config(K, 4, TAU, ATTEMPT, P0, True, 'deterministic'))['makespan'],
                       barrier=simulate_lanes(w['graph'], w['groups'], K, 4, TAU, svd, 'barrier')['makespan'],
                       dynamic=simulate_lanes(w['graph'], w['groups'], K, 4, TAU, svd, 'dynamic')['makespan'])
            for mode in vals:
                d = np.array(vals[mode]) - np.array(vals['lockstep'])
                rows.append(dict(mapping=mp, K=K, balance=bool(K == kavg), mode=mode, seedbase=base,
                                 **stats(vals[mode], trate), det=100 * (det[mode] / trate - 1),
                                 paired_diff=100 * d.mean() / trate,
                                 paired_se=100 * d.std(ddof=1) / math.sqrt(n) / trate))
            print('  async', mp, K, {k: round(100 * (np.mean(v) / trate - 1), 2) for k, v in vals.items()}, flush=True)
    return rows


def part_budget(runs, wr):
    work, rows = workloads(), []
    for mi, mp in enumerate(('JW', 'TT')):
        sizes = work[mp]['sizes']; kavg = math.ceil(sizes.mean() * MU / TAU); rho = kavg / MU
        limit = 100 * (np.maximum(1.0, sizes / (rho * TAU)).mean() - 1)
        for ni, N in enumerate((5, 10, 20, 30, 50, 69, 100, 150, 200)):
            m = np.repeat(sizes, N); t0 = len(m) * TAU
            base = 14000000 + 100000 * mi + 10000 * ni; cfg = Config(kavg, 4); vals = []
            for i in range(runs):
                vals.append(simulate(m, cfg, seed=base + i)['makespan'])
                wr.writerow([f'budget_{mp}_{N}', i, base + i, vals[-1]])
            srt = np.sort(m)[::-1].copy()
            rows.append(dict(mapping=mp, N=N, K=kavg, seedbase=base, **stats(vals, t0),
                             det=100 * (deterministic_runtime(m, kavg, 4, MU, TAU) / t0 - 1),
                             det_sorted=100 * (deterministic_runtime(srt, kavg, 4, MU, TAU) / t0 - 1), limit=limit))
            print('  budget', mp, N, round(rows[-1]['excess'], 2), round(rows[-1]['det'], 2), flush=True)
    return rows


SCENARIOS = {'single stage, late rejection': ((Stage(9, .08),), 0.0),
             'two stages, early rejection': ((Stage(2, .1), Stage(7, .8)), 0.0),
             'three stages, early rejection': ((Stage(1, .5), Stage(3, .2), Stage(5, .8)), 0.0),
             'three stages, reset of 3 cycles': ((Stage(1, .5), Stage(3, .2), Stage(5, .8)), 3.0)}


def part_staged(runs, wr):
    work, rows = workloads(), []
    for mi, mp in enumerate(('JW', 'TT')):
        m = work[mp]['demand']; nT, S = int(m.sum()), len(m); t0 = S * TAU
        for si, (name, (stages, reset)) in enumerate(SCENARIOS.items()):
            mom = staged_moments(stages, reset); mu = mom['mean']
            K = math.ceil(nT * mu / t0); trate = max(t0, nT * mu / K)
            base = 15000000 + 100000 * mi + 10000 * si
            cfg = Config(K, 4, TAU, mu, 1.0, True, 'deterministic')      # carries K, B and tau only
            vals = []
            for i in range(2 * runs):
                sv = sample_staged_fast(stages, (K, nT + 8), base + i, reset)
                vals.append(simulate(m, cfg, services=sv)['makespan'])
                wr.writerow([f'staged_{mp}_{si}', i, base + i, vals[-1]])
            rows.append(dict(mapping=mp, scenario=name, stages=[[s.duration, s.acceptance] for s in stages], reset=reset,
                             mean_latency=mu, latency_cv=100 * math.sqrt(mom['variance']) / mu,
                             acceptance=mom['acceptance'], K=K, seedbase=base, **stats(vals, trate),
                             det=100 * (deterministic_runtime(m, K, 4, mu, TAU) / trate - 1)))
            print('  staged', mp, name, K, round(rows[-1]['excess'], 2), round(rows[-1]['det'], 2), flush=True)
    return rows


def part_external(runs, wr, folder):
    rows = []
    files = sorted(glob.glob(str(folder / '*.u8.xz')))
    if not files:
        raise SystemExit(f'no traces in {folder}; run scripts/fetch_external_traces.py first')
    for fi, f in enumerate(files):
        name = os.path.basename(f)[:-6]
        m = load_trace(f)
        nT, S, peak = int(m.sum()), len(m), int(m.max())
        B = max(4, peak); t0 = S * TAU; K = math.ceil(nT * MU / t0); trate = max(t0, nT * MU / K); rho = K / MU
        ex = lambda t: 100 * (t / trate - 1)
        family = 'qaoa' if name.startswith('qaoa') else name.rsplit('_n', 1)[0]
        row = dict(name=name, family=family, S=S, N=nT, peak=peak, mean=nT / S, K=K, B=B,
                   det=ex(deterministic_runtime(m, K, B, MU, TAU)),
                   det_staggered=ex(deterministic_runtime(m, K, B, MU, TAU, staggered=True)),
                   fluid_lo=ex(fluid_runtime(m, rho, B + K, TAU)), fluid_hi=ex(fluid_runtime(m, rho, B + 1, TAU) + MU),
                   det_sorted=ex(deterministic_runtime(np.sort(m)[::-1].copy(), K, B, MU, TAU)))
        if B + 1 >= rho * TAU:
            row['env_hi'] = ex(histogram_envelope(m, rho, B + 1, TAU)['upper'] + MU)
        n = runs if nT <= 20000 else (16 if nT <= 1100000 else 0)
        if n:
            base = 16000000 + 10000 * fi; cfg = Config(K, B); vals = []
            for i in range(n):
                vals.append(simulate(m, cfg, seed=base + i)['makespan'])
                wr.writerow([f'external_{name}', i, base + i, vals[-1]])
            row.update(seedbase=base, **stats(vals, trate))
        rows.append(row)
        print('  %-44s K=%2d B=%2d det %6.2f sorted %6.2f sim %s' % (name, K, B, row['det'], row['det_sorted'],
              ('%6.2f (n=%d)' % (row['excess'], n)) if n else '   -'), flush=True)
    return rows


def merge():
    out = dict(version='0.3.0'); elapsed = 0.0
    with gzip.open(ROOT / 'data/results/raw_robustness_runs.csv.gz', 'wt', newline='') as raw:
        raw.write('experiment,run,seed,makespan\n')
        for p in PARTS:
            piece = json.loads((TMP / f'{p}.json').read_text())
            out[p] = piece['rows']; elapsed += piece['elapsed_seconds']
            raw.write((TMP / f'{p}.csv').read_text())
    out['elapsed_seconds'] = elapsed
    (ROOT / 'data/results/robustness.json').write_text(json.dumps(out, indent=1) + '\n')
    print('merged', {p: len(out[p]) for p in PARTS}, 'elapsed %.0f s' % elapsed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--external', type=Path, default=ROOT / 'data/external/barc')
    ap.add_argument('--runs', type=int, default=256)
    ap.add_argument('--part', choices=PARTS + ['all', 'merge'], default='all')
    a = ap.parse_args()
    TMP.mkdir(parents=True, exist_ok=True)
    for p in (PARTS if a.part == 'all' else [a.part] if a.part != 'merge' else []):
        t = time.time()
        with open(TMP / f'{p}.csv', 'w', newline='') as f:
            wr = csv.writer(f)
            rows = (part_external(a.runs, wr, a.external) if p == 'external'
                    else {'asynchronous': part_asynchronous, 'budget': part_budget, 'staged': part_staged}[p](a.runs, wr))
        (TMP / f'{p}.json').write_text(json.dumps(dict(rows=rows, elapsed_seconds=time.time() - t)))
        print(f'{p}: {time.time() - t:.0f} s', flush=True)
    if a.part in ('all', 'merge'):
        merge()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Closed-form analysis that accompanies the frozen Monte Carlo study.

Reads  data/results/study.json and data/traces/*.csv
Writes data/results/theory.json and data/results/raw_theory_runs.csv.gz

Everything except two small stochastic batches (order-by-buffer control and
the stall profile) is deterministic and needs no random numbers.
"""
from __future__ import annotations
import argparse, csv, gzip, json, time
from itertools import permutations
from pathlib import Path
import numpy as np
from traceq.model import Config, simulate
from traceq.analysis import balanced_order, summarize
from traceq.theory import (deterministic_runtime, fluid_runtime, fluid_overload_intervals,
                           histogram_envelope, cluster_family, cluster_family_runtimes,
                           minimax_floor)
from traceq.policies import tick_policy, inventory_cap, HOLDING, INJECTION

ROOT = Path(__file__).resolve().parents[1]
TAU, ATTEMPT = 17.0, 9.0


def load_trace(name):
    rows = list(csv.reader(open(ROOT / 'data/traces' / f'{name}.csv')))
    return np.array([int(r[1]) for r in rows[1:]], dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', type=int, default=256, help='realisations per new stochastic batch')
    args = ap.parse_args()
    t_begin = time.time()
    study = json.loads((ROOT / 'data/results/study.json').read_text())
    if study['quick']:
        raise RuntimeError('theory analysis needs the full study, not a smoke run')
    res = {r['id']: r for r in study['results']}
    tr = {'JW': load_trace('hubbard_6x4_JW'), 'TT': load_trace('hubbard_6x4_TT')}
    orders = {k: load_trace(f'TT_order_{k}') for k in ('interleaved', 'shuffled', 'compiled', 'clustered')}
    out = dict(version='0.2.0', source_study_version=study['study_version'])

    # ---- 1. exact separation family --------------------------------------
    fam = []
    for w in (2, 3, 4, 8, 16):
        for c in (2, 3, 4, 8, 16, 32):
            for r in (1, 2, 4, 8, 16, 64):
                a, b = cluster_family(w, c, r)
                cfg = Config(1, w, 1, 1, 1, True, 'deterministic')
                ta, tb = simulate(a, cfg)['makespan'], simulate(b, cfg)['makespan']
                fa, fb = cluster_family_runtimes(w, c, r)
                assert (ta, tb) == (fa, fb), (w, c, r, ta, tb, fa, fb)
                assert deterministic_runtime(b, 1, w, 1., 1.) == tb
                fam.append(dict(w=w, c=c, r=r, spread=ta, cluster=tb,
                                excess_pct=100 * (tb / ta - 1),
                                limit_excess_pct=100 * (1 - 1 / c - 1 / w),
                                minimax=minimax_floor(ta, tb),
                                fluid_bound=fluid_runtime(b, 1., w + 1, 1.)))
    out['family'] = fam

    # ---- 2. closed form against every stored stochastic batch -------------
    def trace_for(rid, r):
        mp = r['mapping']
        if rid.startswith('order_'):
            return orders[rid[6:]]
        if rid.startswith('shuffle_ensemble_'):
            return np.random.default_rng(84000 + int(rid.split('_')[-1])).permutation(tr['TT'])
        if rid.startswith('repeated_'):
            return np.tile(tr[mp], r['repetitions'])
        return tr[mp]

    pred, exact_checks = [], []
    for rid, r in res.items():
        cfg = r['config']
        K, B, p = cfg['producers'], cfg['buffer'], cfg['acceptance']
        mu = ATTEMPT / p
        m = trace_for(rid, r)
        if cfg['law'] == 'deterministic':
            t = deterministic_runtime(m, K, B, mu, TAU, staggered=(r['mode'] == 'staggered'))
            exact_checks.append(dict(id=rid, simulated=r['stats']['mean'], closed_form=t,
                                     equal=bool(abs(t - r['stats']['mean']) < 1e-6)))
            continue
        if cfg['law'] != 'geometric' or r['mode'] is not None:
            continue
        trate, rho = r['trate'], K / mu
        ex = lambda t: 100 * (t / trate - 1)
        det = deterministic_runtime(m, K, B, mu, TAU, prefill=cfg['prefill'])
        row = dict(id=rid, family=rid.split('_')[0], mapping=r['mapping'], K=K, B=B, p=p,
                   n=r['stats']['n'], prefill=cfg['prefill'],
                   sim=r['rate_excess_pct'], sim_lo=ex(r['stats']['mean_ci_low']),
                   sim_hi=ex(r['stats']['mean_ci_high']), det=ex(det),
                   fluid_lo=ex(fluid_runtime(m, rho, B + K, TAU)),
                   fluid_hi=ex(fluid_runtime(m, rho, B + 1, TAU) + mu))
        if B + 1 >= rho * TAU:
            row['env_hi'] = ex(histogram_envelope(m, rho, B + 1, TAU)['upper'] + mu)
            row['env_lo'] = ex(histogram_envelope(m, rho, B + K, TAU)['lower'])
        pred.append(row)
    out['predictor'] = pred
    out['exact_checks'] = exact_checks
    assert all(c['equal'] for c in exact_checks)

    def err_stats(rows):
        e = np.array([abs(x['det'] - x['sim']) for x in rows])
        s = np.array([x['det'] - x['sim'] for x in rows])
        return dict(n=len(rows), median_abs=float(np.median(e)), p90_abs=float(np.quantile(e, .9)),
                    max_abs=float(e.max()), mean_signed=float(s.mean()),
                    inside_bracket=int(sum(x['fluid_lo'] - 1e-9 <= x['det'] <= x['fluid_hi'] + 1e-9 for x in rows)))
    groups = {}
    for x in pred:
        groups.setdefault(x['family'], []).append(x)
    out['predictor_summary'] = {k: err_stats(v) for k, v in groups.items()}
    structured = [x for x in pred if x['family'] not in ('shuffle', 'order')] + \
                 [x for x in pred if x['id'] in ('order_compiled', 'order_clustered')]
    out['predictor_summary']['all'] = err_stats(pred)
    out['predictor_summary']['compiled_and_clustered'] = err_stats(structured)
    out['predictor_summary']['declustered'] = err_stats(
        [x for x in pred if x['family'] == 'shuffle' or x['id'] in ('order_shuffled', 'order_interleaved')])
    big = [x for x in structured if x['det'] >= 5.0]
    out['predictor_summary']['det_excess_at_least_5pct'] = err_stats(big)

    # ---- 3. new stochastic control: four orderings by buffer size ---------
    raw = gzip.open(ROOT / 'data/results/raw_theory_runs.csv.gz', 'wt', newline='')
    wr = csv.writer(raw); wr.writerow(['experiment', 'run', 'seed', 'makespan'])
    ob = []
    for oi, (name, m) in enumerate(orders.items()):
        t0 = len(m) * TAU
        for bi, B in enumerate((4, 8, 16, 32, 64)):
            cfg = Config(11, B)
            base = 11000000 + 100000 * oi + 10000 * bi
            vals = []
            for i in range(args.runs):
                v = simulate(m, cfg, seed=base + i)['makespan']
                vals.append(v); wr.writerow([f'orderbuffer_{name}_{B}', i, base + i, v])
            st = summarize(vals)
            ob.append(dict(order=name, B=B, n=args.runs, seedbase=base,
                           sim=100 * (st['mean'] / t0 - 1), sim_lo=100 * (st['mean_ci_low'] / t0 - 1),
                           sim_hi=100 * (st['mean_ci_high'] / t0 - 1),
                           det=100 * (deterministic_runtime(m, 11, B, 112.5, TAU) / t0 - 1)))
    out['order_buffer'] = ob

    # ---- 4. where the delay accrues: stall profile along the schedule ----
    prof = {}
    for mi, (mp, K) in enumerate((('JW', 10), ('TT', 11))):
        m = tr[mp]; S = len(m); grid = np.arange(S) * TAU
        _, st_det = deterministic_runtime(m, K, 4, 112.5, TAU, return_starts=True)
        acc = np.zeros(S); base = 12000000 + 100000 * mi
        for i in range(args.runs):
            x = simulate(m, Config(K, 4), seed=base + i)
            acc += x['starts'] - grid; wr.writerow([f'profile_{mp}', i, base + i, x['makespan']])
        step = 4
        fam_hi = fluid_overload_intervals(m, K / 112.5, 4 + 1, TAU)
        fam_lo = fluid_overload_intervals(m, K / 112.5, 4 + K, TAU)
        prof[mp] = dict(K=K, B=4, n=args.runs, seedbase=base, stride=step,
                        det=(st_det - grid)[::step].round(3).tolist(),
                        sim=(acc / args.runs)[::step].round(3).tolist(),
                        intervals_C_B_plus_1=[[f['start'], f['stop'], round(f['gain'], 3)] for f in fam_hi],
                        intervals_C_B_plus_K=[[f['start'], f['stop'], round(f['gain'], 3)] for f in fam_lo])
    raw.close()
    out['profile'] = prof

    # ---- 5. what the histogram alone allows, across K ---------------------
    env = []
    for mp, m in tr.items():
        il, srt = balanced_order(m), np.sort(m)[::-1].copy()
        for K in range(6, 25):
            r = res[f'capacity_{mp}_{K}']; trate = r['trate']; rho = K / 112.5
            ex = lambda t: 100 * (t / trate - 1)
            env.append(dict(mapping=mp, K=K, B=4, sim=r['rate_excess_pct'],
                            upper_hi=ex(histogram_envelope(m, rho, 5, TAU)['upper'] + 112.5),
                            upper_lo=ex(histogram_envelope(m, rho, 4 + K, TAU)['upper']),
                            lower=ex(histogram_envelope(m, rho, 4 + K, TAU)['lower']),
                            det_compiled=ex(deterministic_runtime(m, K, 4, 112.5, TAU)),
                            det_sorted=ex(deterministic_runtime(srt, K, 4, 112.5, TAU)),
                            det_interleaved=ex(deterministic_runtime(il, K, 4, 112.5, TAU))))
    out['envelope'] = env

    # ---- 6. exhaustive rearrangement check on small multisets -------------
    rng = np.random.default_rng(20260918)
    cases = fluid_viol = disc_not_max = orderings = 0
    worst_short = 0.0
    while cases < 600:
        S = int(rng.integers(3, 8)); B = int(rng.integers(1, 6)); K = int(rng.integers(1, 9))
        m = rng.integers(0, B + 1, S)
        if m.sum() == 0:
            continue
        mu = float(rng.choice([1., 2., 3.5, 9.])); tau = float(rng.choice([.5, 1., 2., 5.])); rho = K / mu
        if B + 1 < rho * tau:
            continue
        perms = set(permutations(m.tolist())); cases += 1; orderings += len(perms)
        up = histogram_envelope(m, rho, B + 1, tau)['upper']
        if max(fluid_runtime(q, rho, B + 1, tau) for q in perms) > up + 1e-9:
            fluid_viol += 1
        dv = [deterministic_runtime(q, K, B, mu, tau) for q in perms]
        ds = max(deterministic_runtime(sorted(m), K, B, mu, tau),
                 deterministic_runtime(sorted(m, reverse=True), K, B, mu, tau))
        if max(dv) > ds + 1e-9:
            disc_not_max += 1; worst_short = max(worst_short, (max(dv) - ds) / mu)
        assert max(dv) <= up + mu + 1e-9
    out['enumeration'] = dict(multisets=cases, orderings=orderings, fluid_violations=fluid_viol,
                              discrete_sorted_not_maximal=disc_not_max,
                              worst_shortfall_in_latencies=worst_short)

    # ---- 7. alternative holding / injection policies on the family -------
    pol = []
    for (w, c, r) in ((4, 2, 16), (4, 8, 16), (8, 4, 8)):
        a, b = cluster_family(w, c, r)
        for h in HOLDING:
            for inj in INJECTION:
                cap = inventory_cap(b, 1, w, h, inj)
                pol.append(dict(w=w, c=c, r=r, holding=h, injection=inj, inventory_cap=cap,
                                spread=tick_policy(a, 1, w, 1, 1, h, inj),
                                cluster=tick_policy(b, 1, w, 1, 1, h, inj),
                                bound=fluid_runtime(b, 1., cap, 1.)))
    out['policies'] = pol

    # ---- 8. the closed form as a screen for the design grid ---------------
    sel = json.loads((ROOT / 'data/results/selected_candidates.json').read_text())
    scr = []
    for cnd in sel:
        m = tr[cnd['mapping']]; t0 = len(m) * TAU
        kd = next(K for K in range(6, 40)
                  if deterministic_runtime(m, K, cnd['B'], 112.5, TAU) <= 1.02 * t0)
        scr.append(dict(mapping=cnd['mapping'], B=cnd['B'], K_closed_form=kd, K_screened=cnd['K']))
    out['screen'] = scr

    out['elapsed_seconds'] = time.time() - t_begin
    (ROOT / 'data/results/theory.json').write_text(json.dumps(out, indent=1) + '\n')
    print('predictor batches', len(pred), '| exact deterministic checks', len(exact_checks))
    for k, v in out['predictor_summary'].items():
        print('  %-26s n=%3d median|err|=%.3f p90=%.3f max=%.3f' % (k, v['n'], v['median_abs'], v['p90_abs'], v['max_abs']))
    print('enumeration', out['enumeration'])
    print('screen', scr)
    print('DONE in %.1f s' % out['elapsed_seconds'])


if __name__ == '__main__':
    main()

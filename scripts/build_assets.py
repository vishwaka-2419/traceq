#!/usr/bin/env python3
"""Generate manuscript tables and numeric macros from study.json, theory.json and robustness.json.

Figures are produced separately by scripts/make_figures.py.
"""
from __future__ import annotations
import json, re
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]


def main():
    s = json.loads((ROOT / 'data/results/study.json').read_text())
    if s['quick']:
        raise RuntimeError('Refusing to typeset the manuscript from quick-mode data')
    th = json.loads((ROOT / 'data/results/theory.json').read_text())
    rb = json.loads((ROOT / 'data/results/robustness.json').read_text())
    d = {r['id']: r for r in s['results']}
    P = {r['id']: r for r in th['predictor']}
    dest = ROOT / 'manuscript' if (ROOT / 'manuscript').is_dir() else ROOT / 'generated'
    dest.mkdir(exist_ok=True)
    base = {m: d['baseline_' + m] for m in ['JW', 'TT']}
    meanK = {m: min(r['config']['producers'] for r in s['results']
                    if r['id'].startswith('capacity_' + m + '_') and r['stats']['mean'] <= 1.02 * r['t0'])
             for m in ['JW', 'TT']}
    orders = [d['order_' + k] for k in ['shuffled', 'interleaved', 'compiled', 'clustered']]
    shuffle = [r['rate_excess_pct'] for r in s['results'] if r['id'].startswith('shuffle_ensemble_')]
    count = re.search(r'(\d+) passed', (ROOT / 'data/results/test_report.txt').read_text()).group(1)
    ps = th['predictor_summary']
    env = {(e['mapping'], e['K']): e for e in th['envelope']}
    frac = lambda e: 100 * e['det_compiled'] / e['det_sorted']
    fr_all = [frac(e) for e in th['envelope'] if 8 <= e['K'] <= 17]
    ob = {(r['order'], r['B']): r for r in th['order_buffer']}
    prem = lambda o, b: ob[(o, b)]['sim'] - ob[(o, b)]['det']
    fam = {(f['w'], f['c'], f['r']): f for f in th['family']}
    new_runs = sum(r['n'] for r in th['order_buffer']) + sum(v['n'] for v in th['profile'].values())
    f2 = lambda x: f'{x:.2f}'
    macros = {
        'JWExcess': f2(base['JW']['rate_excess_pct']), 'TTExcess': f2(base['TT']['rate_excess_pct']),
        'OrderSpan': f2(max(x['rate_excess_pct'] for x in orders) - min(x['rate_excess_pct'] for x in orders)),
        'JWFloor': f"{base['JW']['t0']:,.0f}", 'TTFloor': f"{base['TT']['t0']:,.0f}",
        'KMeanJW': str(meanK['JW']), 'KMeanTT': str(meanK['TT']),
        'KFactorJW': f2(meanK['JW'] / 10), 'KFactorTT': f2(meanK['TT'] / 11),
        'ShuffleLow': f2(min(shuffle)), 'ShuffleHigh': f2(max(shuffle)),
        'TTBufferFour': f2(d['buffer_TT_4']['rate_excess_pct']),
        'TTBufferThirtyTwo': f2(d['buffer_TT_32']['rate_excess_pct']),
        'TTAgeFour': f"{d['buffer_TT_4']['mean_residency']:.1f}",
        'TTAgeThirtyTwo': f"{d['buffer_TT_32']['mean_residency']:.1f}",
        'HighPJW': f2(d['acceptance_JW_0.75']['rate_excess_pct']), 'HighPTT': f2(d['acceptance_TT_0.75']['rate_excess_pct']),
        'RepeatEightJW': f2(d['repeated_JW_8']['rate_excess_pct']), 'RepeatEightTT': f2(d['repeated_TT_8']['rate_excess_pct']),
        'JWDetExcess': f2(d['law_JW_deterministic']['rate_excess_pct']),
        'TTDetExcess': f2(d['law_TT_deterministic']['rate_excess_pct']),
        'TestCount': count, 'TotalRuns': f"{sum(r['stats']['n'] for r in s['results']):,}",
        # ---- closed-form analysis
        'TTOrderCompiled': f2(d['order_compiled']['rate_excess_pct']),
        'TTOrderClustered': f2(d['order_clustered']['rate_excess_pct']),
        'TTOrderShuffled': f2(d['order_shuffled']['rate_excess_pct']),
        'TTOrderInterleaved': f2(d['order_interleaved']['rate_excess_pct']),
        'DetInterleaved': f2(P['order_interleaved']['det']), 'DetShuffled': f2(P['order_shuffled']['det']),
        'DetCompiled': f2(P['order_compiled']['det']), 'DetClustered': f2(P['order_clustered']['det']),
        'FluidLoJW': f2(P['baseline_JW']['fluid_lo']), 'FluidHiJW': f2(P['baseline_JW']['fluid_hi']),
        'FluidLoTT': f2(P['baseline_TT']['fluid_lo']), 'FluidHiTT': f2(P['baseline_TT']['fluid_hi']),
        'EnvHiJW': f2(P['baseline_JW']['env_hi']), 'EnvHiTT': f2(P['baseline_TT']['env_hi']),
        'SortedJW': f2(env[('JW', 10)]['det_sorted']), 'SortedTT': f2(env[('TT', 11)]['det_sorted']),
        'WorstFracJW': f"{frac(env[('JW', 10)]):.0f}", 'WorstFracTT': f"{frac(env[('TT', 11)]):.0f}",
        'WorstFracMin': f"{min(fr_all):.0f}", 'WorstFracMax': f"{max(fr_all):.0f}",
        'PredBatches': str(ps['all']['n']),
        'PredStructN': str(ps['compiled_and_clustered']['n']),
        'PredStructMedian': f2(ps['compiled_and_clustered']['median_abs']),
        'PredStructPninety': f2(ps['compiled_and_clustered']['p90_abs']),
        'PredStructMax': f2(ps['compiled_and_clustered']['max_abs']),
        'PredBigN': str(ps['det_excess_at_least_5pct']['n']),
        'PredBigMedian': f2(ps['det_excess_at_least_5pct']['median_abs']),
        'PredBigMax': f2(ps['det_excess_at_least_5pct']['max_abs']),
        'PredDeclN': str(ps['declustered']['n']),
        'PredDeclMedian': f2(ps['declustered']['median_abs']), 'PredDeclMax': f2(ps['declustered']['max_abs']),
        'PremInterFour': f2(prem('interleaved', 4)), 'PremInterEight': f2(prem('interleaved', 8)),
        'PremInterSixteen': f2(prem('interleaved', 16)), 'PremInterThirtyTwo': f2(prem('interleaved', 32)),
        'PremShufFour': f2(prem('shuffled', 4)), 'PremShufThirtyTwo': f2(prem('shuffled', 32)),
        'PremCompFour': f2(prem('compiled', 4)), 'PremCompMax': f2(max(prem('compiled', b) for b in (4, 8, 16, 32, 64))),
        'NIntervalsJW': str(len(th['profile']['JW']['intervals_C_B_plus_1'])),
        'NIntervalsTT': str(len(th['profile']['TT']['intervals_C_B_plus_1'])),
        'EnumMultisets': f"{th['enumeration']['multisets']:,}", 'EnumOrderings': f"{th['enumeration']['orderings']:,}",
        'TheoryRuns': f'{new_runs:,}', 'TheoryBatches': str(len(th['order_buffer']) + len(th['profile'])),
        'ScreenEqual': str(sum(x['K_closed_form'] == x['K_screened'] for x in th['screen'])),
        'ScreenOneBelow': str(sum(x['K_screened'] - x['K_closed_form'] == 1 for x in th['screen'])),
        'FamFourTwo': f"{fam[(4, 2, 64)]['excess_pct']:.1f}", 'FamFourThirtyTwo': f"{fam[(4, 32, 64)]['excess_pct']:.1f}",
        'FamSixteenThirtyTwo': f"{fam[(16, 32, 64)]['excess_pct']:.1f}",
        'FamSixteenFloor': f"{fam[(16, 32, 64)]['minimax']:.3f}",
    }
    asy = {(x['mapping'], x['K'], x['mode']): x for x in rb['asynchronous']}
    bal = {(x['mapping'], x['mode']): x for x in rb['asynchronous'] if x['balance']}
    bud = {(x['mapping'], x['N']): x for x in rb['budget']}
    ext = rb['external']; exs = [x for x in ext if 'excess' in x]
    gap = np.array([abs(x['det'] - x['excess']) for x in exs])
    fam_rows = lambda f: [x for x in ext if x['family'] == f]
    fsim = lambda f: [x['excess'] for x in fam_rows(f) if 'excess' in x]
    fdet = lambda f: [x['det'] for x in fam_rows(f)]
    small = [x['excess'] for x in exs if x['family'] in ('adder', 'modular_multiplier')]
    macros.update({
        'AsyncRuns': str(bal[('JW', 'lockstep')]['n']),
        'AsyncJWLock': f2(bal[('JW', 'lockstep')]['excess']), 'AsyncJWBarrier': f2(bal[('JW', 'barrier')]['excess']),
        'AsyncJWDynamic': f2(bal[('JW', 'dynamic')]['excess']), 'AsyncTTLock': f2(bal[('TT', 'lockstep')]['excess']),
        'AsyncTTBarrier': f2(bal[('TT', 'barrier')]['excess']), 'AsyncTTDynamic': f2(bal[('TT', 'dynamic')]['excess']),
        'AsyncBarrierMaxDiff': f2(max(abs(x['paired_diff']) for x in rb['asynchronous'] if x['mode'] == 'barrier')),
        'AsyncJWDynDiff': f"{bal[('JW', 'dynamic')]['paired_diff']:+.2f}", 'AsyncTTDynDiff': f"{bal[('TT', 'dynamic')]['paired_diff']:+.2f}".replace('-', '$-$'),
        'AsyncTTBarrierDet': f2(bal[('TT', 'barrier')]['det']), 'AsyncTTDynamicDet': f2(bal[('TT', 'dynamic')]['det']),
        'BudgetJWFive': f2(bud[('JW', 5)]['excess']), 'BudgetJWTwoHundred': f2(bud[('JW', 200)]['excess']),
        'BudgetTTFive': f2(bud[('TT', 5)]['excess']), 'BudgetTTTwoHundred': f2(bud[('TT', 200)]['excess']),
        'BudgetJWLimit': f2(bud[('JW', 5)]['limit']), 'BudgetTTLimit': f2(bud[('TT', 5)]['limit']),
        'BudgetMaxGap': f2(max(abs(x['det'] - x['excess']) for x in rb['budget'])),
        'StagedMin': f2(min(x['excess'] for x in rb['staged'])), 'StagedMax': f2(max(x['excess'] for x in rb['staged'])),
        'StagedMaxGap': f2(max(abs(x['det'] - x['excess']) for x in rb['staged'])),
        'ExtTraces': str(len(ext)), 'ExtSimulated': str(len(exs)),
        'ExtAtLeastFive': str(sum(x['excess'] >= 5 for x in exs)), 'ExtAtLeastTen': str(sum(x['excess'] >= 10 for x in exs)),
        'ExtMedianGap': f2(np.median(gap)), 'ExtPninetyGap': f2(np.quantile(gap, .9)), 'ExtMaxGap': f2(gap.max()),
        'ExtQFTDetMin': f2(min(fdet('qft'))), 'ExtQFTDetMax': f2(max(fdet('qft'))), 'ExtQFTSim': f2(fsim('qft')[0]),
        'ExtQAOADetMedian': f2(np.median(fdet('qaoa'))), 'ExtQAOASimMedian': f2(np.median(fsim('qaoa'))),
        'ExtQAOASimMin': f2(min(fsim('qaoa'))), 'ExtQAOASimMax': f2(max(fsim('qaoa'))),
        'ExtCLADetMin': f2(min(fdet('cla_adder'))), 'ExtCLADetMax': f2(max(fdet('cla_adder'))),
        'ExtCLASimMin': f2(min(fsim('cla_adder'))), 'ExtCLASimMax': f2(max(fsim('cla_adder'))),
        'ExtRippleDetMax': f2(max(fdet('adder') + fdet('modular_multiplier') + fdet('multiplier'))),
        'ExtSmallSimMin': f2(min(small)), 'ExtSmallSimMax': f2(max(small)),
        'ExtMultSimMin': f2(min(fsim('multiplier'))), 'ExtMultSimMax': f2(max(fsim('multiplier'))),
        'ExtFracMax': f"{100 * max(x['det'] / x['det_sorted'] for x in ext):.0f}",
        'ExtSortedMin': f2(min(x['det_sorted'] for x in ext)), 'ExtSortedMax': f2(max(x['det_sorted'] for x in ext)),
        'RobustRuns': f"{sum(x['n'] for x in rb['asynchronous']) + sum(x['n'] for x in rb['budget']) + sum(x['n'] for x in rb['staged']) + sum(x.get('n', 0) for x in ext):,}",
    })
    (dest / 'generated_numbers.tex').write_text(
        '% Auto-generated from study.json and theory.json; do not edit.\n'
        + ''.join('\\newcommand{\\' + k + '}{' + v + '}\n' for k, v in macros.items()))
    (ROOT / 'data/results/headline_numbers.json').write_text(json.dumps(macros, indent=2) + '\n')

    def table(name, spec, header, rows, caption, label, size='\\small'):
        txt = '\\begin{table}[htbp]\n\\centering' + size + '\n\\begin{tabular}{' + spec + '}\n\\toprule\n'
        txt += ' & '.join(header) + '\\\\\\midrule\n'
        txt += ''.join(' & '.join(map(str, row)) + '\\\\\n' for row in rows)
        txt += '\\bottomrule\n\\end{tabular}\n\\caption{' + caption + '}\n\\label{' + label + '}\n\\end{table}\n'
        (dest / (name + '.tex')).write_text(txt)

    rows = []
    for m in ['JW', 'TT']:
        sig = base[m]['signature']; hist = ', '.join(f'{a}: {b:,}' for a, b in sig['histogram'].items())
        rows.append([m, f"{sig['steps']:,}", f"{sig['states']:,}", str(sig['peak']), f"{sig['mean']:.4f}", hist])
    table('table_workloads', 'lrrrrp{0.37\\linewidth}', ['Mapping', '$S$', '$N_T$', 'Peak', 'Mean', 'Histogram $m$: step count'], rows,
          'Workload signatures. Both traces use 69 sequential state requests per Pauli rotation and the same declared footprint and injection budgets.', 'tab:workloads')
    rows = []
    for m in ['JW', 'TT']:
        r = base[m]; ss = r['stats']; ci = [100 * (ss[k] / r['trate'] - 1) for k in ['mean_ci_low', 'mean_ci_high']]
        rows.append([m, r['config']['producers'], f2(r['rate_excess_pct']), f'[{ci[0]:.2f}, {ci[1]:.2f}]',
                     f2(P['baseline_' + m]['det']), f"{100 * ss['cv']:.3f}", f"{ss['q95'] / r['t0']:.4f}", meanK[m]])
    table('table_baseline', 'lrrrrrrr', ['Mapping', '$K_{\\rm avg}$', 'Excess (\\%)', '95\\% mean CI', 'Closed form (\\%)', 'CV (\\%)', '$T_{95}/T_0$', '$K_{\\rm mean}$'], rows,
          'Balance-point results, 4,096 realisations per mapping. Excess and its interval are percentage points above $T_{\\rm rate}=T_0$. The closed form is the exact deterministic runtime of \\cref{thm:maxplus}. The last column is the smallest tested integer count with empirical mean runtime at most $1.02T_0$, from the separate 512-run capacity sweep.', 'tab:baseline')
    rows = []
    for r in orders:
        st = r['stats']; lo = 100 * (st['mean_ci_low'] / r['trate'] - 1); hi = 100 * (st['mean_ci_high'] / r['trate'] - 1)
        rows.append([r['id'][6:].capitalize(), f2(r['rate_excess_pct']), f'[{lo:.2f}, {hi:.2f}]', f2(P[r['id']]['det']),
                     f"{100 * st['cv']:.3f}", f"{r['fluid']['capacity']:.1f}"])
    table('table_order', 'lrrrrr', ['Order', 'Excess (\\%)', '95\\% mean CI', 'Closed form (\\%)', 'CV (\\%)', '$\\mathcal E_{K/\\mu}$'], rows,
          'Identical-histogram TT orderings at $K=11$, $B=4$, with 1,024 realisations each. The closed form is the exact deterministic runtime; $\\mathcal E_{K/\\mu}$ is the largest interval excursion of \\eqref{eq:excursion}, in state units.', 'tab:order')
    rows = []
    for B in [4, 8, 12, 16, 24, 32, 64]:
        r = d[f'buffer_TT_{B}']
        rows.append([B, 11 + B, f2(r['rate_excess_pct']), f2(P[f'buffer_TT_{B}']['det']), f"{r['mean_residency']:.1f}", f"{100 * r['mean_blocked_fraction']:.2f}"])
    table('table_buffer', 'rrrrrr', ['$B$', '$K+B$', 'Excess (\\%)', 'Closed form (\\%)', 'Residency (cycles)', 'Blocked (\\%)'], rows,
          'TT storage sweep at $K=11$, with 512 realisations per point. Residency is the mean over consumed states and includes holding at blocked producers; the last column is the blocked share of producer patch-time. Prefilled states have zero recorded age within the timed window.', 'tab:buffer')
    scr = {(x['mapping'], x['B']): x['K_closed_form'] for x in th['screen']}
    rows = []
    for r in [r for r in s['results'] if 'validation' in r]:
        c = r['config']; st = r['stats']; v = r['validation']
        rows.append([r['mapping'], f"({c['producers']}, {c['buffer']})", scr[(r['mapping'], c['buffer'])], c['producers'] + c['buffer'],
                     f"{st['violations']}/{st['n']}", f"{100 * v['violation_ucb_family']:.3f}", f"{st['q95'] / r['t0']:.4f}", 'Yes' if v['certified'] else 'No'])
    table('table_validation', 'lrrrrrrr', ['Mapping', '$(K,B)$', '$K_{\\rm cf}$', '$K+B$', 'Misses/$n$', 'Upper bound (\\%)', '$T_{95}/T_0$', 'Pass'], rows,
          'Held-out validation of eight preselected candidates. $K_{\\rm cf}$ is the smallest count whose closed-form deterministic runtime is at most $1.02T_0$ at that $B$. Upper bounds have simultaneous coverage of at least 95\\% across this family; a pass requires an upper bound of at most 5\\% for missing $1.02T_0$. Each bound is conditional on the stated model and is not a physical-device guarantee.', 'tab:validation')
    labels = {'geometric': 'Independent geometric', 'deterministic': 'Fixed, aligned', 'staggered': 'Fixed, staggered',
              'two_point': 'Independent two-point', 'common_slowdown': 'Run-common slowdown'}
    rows = []
    for mp in ['JW', 'TT']:
        for law, label in labels.items():
            r = d[f'law_{mp}_{law}']; st = r['stats']
            rows.append([mp, label, st['n'], f2(r['rate_excess_pct']), f"{st['q95'] / r['t0']:.4f}", f"{100 * st['cv']:.3f}"])
    table('table_laws', 'llrrrr', ['Mapping', 'Service law and initial phase', '$n$', 'Excess (\\%)', '$T_{95}/T_0$', 'CV (\\%)'], rows,
          "Controls with the same nominal mean latency at each mapping's balance count and $B=4$. Fixed-service rows are deterministic and agree with the closed form to machine precision. The common-environment model is a synthetic correlation stress test and is not used for the provisioning certificates.", 'tab:laws')
    rows = []
    for mp in ['JW', 'TT']:
        for key, label, rep in [(f'baseline_{mp}', 'Prefilled', 1), (f'cold_{mp}', 'Cold', 1), (f'repeated_{mp}_8', 'Prefilled', 8)]:
            r = d[key]
            rows.append([mp, label, rep, r['stats']['n'], f2(r['rate_excess_pct']), f2(P[key]['det']), f"{100 * r['stats']['cv']:.3f}"])
    table('table_startup', 'llrrrrr', ['Mapping', 'Initial storage', 'Segments', '$n$', 'Excess (\\%)', 'Closed form (\\%)', 'CV (\\%)'], rows,
          'Start-up and repetition controls. One prefill is used per concatenated run. Batches have independent seeds, so comparisons are not paired confidence intervals.', 'tab:startup')

    # ---- new tables
    def prow(label, k):
        v = ps[k]
        return [label, v['n'], f2(v['median_abs']), f2(v['p90_abs']), f2(v['max_abs']), f"{v['mean_signed']:+.2f}"]
    table('table_predictor', 'p{0.43\\linewidth}rrrrr', ['Batches', '$n$', 'Median', '90th pct.', 'Max', 'Mean signed'],
          [prow('Compiled and clustered orders', 'compiled_and_clustered'),
           prow('\\quad of which closed-form excess $\\ge5\\%$', 'det_excess_at_least_5pct'),
           prow('Shuffled and interleaved orders', 'declustered')],
          'Absolute difference, in percentage points of $T_{\\rm rate}$, between the exact deterministic runtime of \\cref{thm:maxplus} and the simulated mean under geometric service, over all ' + macros['PredBatches'] + ' stored geometric-service batches. The signed column is closed form minus simulation.', 'tab:predictor')
    names = {'baseline': 'Balance point', 'capacity': 'Capacity sweep, $B=4$', 'buffer': 'Storage sweep', 'grid': 'Joint $(K,B)$ screen',
             'validation': 'Held-out validation', 'acceptance': 'Acceptance sweep', 'repeated': 'Concatenated traces', 'cold': 'Cold start',
             'law': 'Geometric law control', 'order': 'Four TT orderings', 'shuffle': 'Shuffle ensemble'}
    table('table_predictor_family', 'lrrrrr', ['Experiment family', '$n$', 'Median', '90th pct.', 'Max', 'Mean signed'],
          [prow(names[k], k) for k in names],
          'The comparison of Table~\\ref{M-tab:predictor} of the main text resolved by experiment family. All entries are percentage points of $T_{\\rm rate}$.', 'tab:predictorfamily')
    rows = []
    for o in ('interleaved', 'shuffled', 'compiled', 'clustered'):
        for B in (4, 8, 16, 32, 64):
            r = ob[(o, B)]
            rows.append([o.capitalize() if B == 4 else '', B, f"{r['sim']:.2f}", f"[{r['sim_lo']:.2f}, {r['sim_hi']:.2f}]", f"{r['det']:.2f}", f"{r['sim'] - r['det']:+.2f}"])
    table('table_orderbuffer', 'lrrrrr', ['Order', '$B$', 'Simulated (\\%)', '95\\% mean CI', 'Closed form (\\%)', 'Difference (pp)'], rows,
          'Stochastic premium by ordering and storage: TT histogram, $K=11$, geometric service, ' + str(th['order_buffer'][0]['n']) + ' realisations per row with seeds disjoint from the main study.', 'tab:orderbuffer', size='\\footnotesize')
    hold = {'hold': 'Blocking after service', 'reserve': 'Reservation before service', 'discard': 'Discard on full'}
    rows = []
    for x in th['policies']:
        if (x['w'], x['c']) in ((4, 2), (4, 8)):
            rows.append([f"({x['w']}, {x['c']}, {x['r']})" if (x['holding'], x['injection']) == ('hold', 'atomic') else '',
                         hold[x['holding']], x['injection'].capitalize(), x['inventory_cap'], x['spread'], x['cluster'], f"{x['bound']:.0f}"])
    table('table_policies', 'lllrrrr', ['$(w,c,r)$', 'Holding rule', 'Injection', '$I_{\\max}$', '$T_{\\rm spread}$', '$T_{\\rm cluster}$', 'Bound'], rows,
          'Runtimes of the equal-histogram pair of Theorem~\\ref{M-thm:order} of the main text under six combinations of holding and injection rules ($K=1$, $B=w$, unit latency and step time), with the policy-independent lower bound of Theorem~\\ref{M-thm:policy} evaluated at each rule\'s inventory cap.', 'tab:policies', size='\\footnotesize')
    mode_name = {'lockstep': 'Lockstep (main model)', 'barrier': 'Asynchronous in groups', 'dynamic': 'Dynamic admission'}
    rows = []
    for mp in ('JW', 'TT'):
        for mode in ('lockstep', 'barrier', 'dynamic'):
            x = bal[(mp, mode)]
            diff = '--' if mode == 'lockstep' else f"${x['paired_diff']:+.2f}\\pm{x['paired_se']:.2f}$"
            rows.append([mp if mode == 'lockstep' else '', mode_name[mode], f2(x['excess']), f"[{x['lo']:.2f}, {x['hi']:.2f}]", diff, f2(x['det'])])
    table('table_async', 'llrrrr', ['Map.', 'Progress rule', 'Excess (\\%)', '95\\% CI', 'Diff. (pp)', 'Fixed $\\mu$ (\\%)'], rows,
          'Lockstep against asynchronous progress (within the groups of the static schedule, and fully asynchronous with dynamic admission) at mean-rate balance and $B=4$, ' + macros['AsyncRuns'] + ' realisations per row with common random numbers across the three rules. The difference is the paired mean difference to lockstep with its standard error. The last column is a single run with fixed latency $\\mu$; for the asynchronous rules it depends on the production phase and is shown for completeness.', 'tab:async')
    rows = []
    for mp in ('JW', 'TT'):
        for K in sorted({x['K'] for x in rb['asynchronous'] if x['mapping'] == mp}):
            x = [asy[(mp, K, mo)] for mo in ('lockstep', 'barrier', 'dynamic')]
            rows.append([mp, K, x[0]['n'], f2(x[0]['excess']), f2(x[1]['excess']), f"${x[1]['paired_diff']:+.2f}\\pm{x[1]['paired_se']:.2f}$",
                         f2(x[2]['excess']), f"${x[2]['paired_diff']:+.2f}\\pm{x[2]['paired_se']:.2f}$"])
    table('table_async_sweep', 'lrrrrrrr', ['Mapping', '$K$', '$n$', 'Lockstep', 'Within groups', 'Difference', 'Dynamic', 'Difference'], rows,
          'Asynchronous progress across production counts ($B=4$, geometric service, excess over $T_{\\rm rate}$ in per cent, paired differences to lockstep in percentage points). A negative excess under dynamic admission means that the dynamic schedule finishes before the supply-unconstrained time of the static schedule.', 'tab:asyncsweep', size='\\footnotesize')
    rows = []
    for N in sorted({x['N'] for x in rb['budget']}):
        j, t = bud[('JW', N)], bud[('TT', N)]
        rows.append([N, f2(j['excess']), f2(j['det']), f2(j['det_sorted']), f2(t['excess']), f2(t['det']), f2(t['det_sorted'])])
    table('table_budget', 'rrrrrrr', ['$N_{\\rm synth}$', 'JW sim.', 'JW closed form', 'JW sorted', 'TT sim.', 'TT closed form', 'TT sorted'], rows,
          'Synthesis-chain length: every rotation requests $N_{\\rm synth}$ sequential states; $K=K_{\\rm avg}$ and $B=4$ throughout; ' + str(rb['budget'][0]['n']) + ' realisations per point. All entries are excess over $T_{\\rm rate}$ in per cent. The sorted column is the closed form for the descending arrangement of the same histogram.', 'tab:budget', size='\\footnotesize')
    rows = []
    for x in rb['staged']:
        st = ', '.join(f"({a:g}, {b:g})" for a, b in x['stages'])
        rows.append([x['mapping'], st + (f"; reset {x['reset']:g}" if x['reset'] else ''), f"{x['mean_latency']:.2f}", f"{x['latency_cv']:.0f}",
                     x['K'], f2(x['excess']), f"[{x['lo']:.2f}, {x['hi']:.2f}]", f2(x['det'])])
    table('table_staged', 'l>{\\raggedright\\arraybackslash}p{0.27\\linewidth}rrrrrr', ['Map.', 'Stages (cycles, acceptance)', '$\\mu$', 'CV', '$K_{\\rm avg}$', 'Excess', '95\\% CI', 'Closed'], rows,
          'Stage-resolved service laws with end-to-end acceptance $0.08$ and a nine-cycle successful attempt; latency CV, excess and closed form in per cent. A failed stage restarts the attempt. $K_{\\rm avg}$ is recomputed from the mean latency $\\mu$ of each law; ' + str(rb['staged'][0]['n']) + ' realisations per row; $B=4$. These are declared scenarios and not calibrated protocols.', 'tab:staged', size='\\footnotesize')
    fam_name = {'qft': 'QFT', 'qaoa': 'QAOA MaxCut', 'cla_adder': 'Carry-lookahead adder', 'adder': 'Ripple-carry adder',
                'modular_multiplier': 'Modular multiplier', 'multiplier': 'Multiplier'}
    rng = lambda v: f"{min(v):.1f}--{max(v):.1f}" if len(v) > 1 else f"{v[0]:.1f}"
    rows = []
    for f in ('qft', 'qaoa', 'cla_adder', 'multiplier', 'adder', 'modular_multiplier'):
        v = fam_rows(f)
        irng = lambda k: (f"{min(x[k] for x in v)}--{max(x[k] for x in v)}" if min(x[k] for x in v) < max(x[k] for x in v) else str(v[0][k]))
        rows.append([fam_name[f], len(v), irng('K'), irng('B'),
                     rng(fdet(f)), rng(fsim(f)), rng([x['det_sorted'] for x in v])])
    table('table_external', 'lrrrrrr', ['Family', 'Traces', '$K_{\\rm avg}$', '$B$', 'Closed form', 'Simulated', 'Sorted order'], rows,
          'Independently produced demand traces~\\cite{ye2026} at mean-rate balance, with $B=\\max\\{4,\\max_sm_s\\}$: ranges of the excess over $T_{\\rm rate}$, in per cent, within each family for the exact deterministic runtime of the given order, the simulated mean under geometric service, and the deterministic runtime of the sorted, worst ordering of the same histogram. Two QFT traces with more than $10^6$ states were not simulated.', 'tab:external')
    ordered = sorted(ext, key=lambda x: (list(fam_name).index(x['family']), x['N']))
    short = lambda n: n.replace('qaoa_maxcut_', 'qaoa ').replace('erdos_renyi_dense', 'ER').replace('random_3_regular', '3reg').replace('modular_multiplier', 'modmult').replace('_', ' ')
    for part, chunk in enumerate((ordered[:32], ordered[32:])):
        rows = [[short(x['name']), f"{x['S']:,}", f"{x['N']:,}", x['peak'], x['K'], x['B'], f2(x['det']), f"[{x['fluid_lo']:.1f}, {x['fluid_hi']:.1f}]",
                 f2(x['det_sorted']), (f2(x['excess']) if 'excess' in x else '--'), (x['n'] if 'excess' in x else '--')] for x in chunk]
        table(f'table_external_all_{part + 1}', 'lrrrrrrrrrr', ['Trace', '$S$', '$N_T$', 'Peak', '$K$', '$B$', 'Closed', 'Fluid bracket', 'Sorted', 'Sim.', '$n$'], rows,
              ('All ' + macros['ExtTraces'] + ' external traces, part ' + str(part + 1) + ' of 2. Excess over $T_{\\rm rate}$ in per cent at $K=K_{\\rm avg}$ and $B=\\max\\{4,\\text{peak}\\}$. Closed: exact deterministic runtime; bracket: \\cref{M-cor:sandwich}; sorted: deterministic runtime of the descending arrangement; Sim.: mean of $n$ geometric-service realisations.').replace('\\cref{M-cor:sandwich}', 'Corollary~\\ref{M-cor:sandwich} of the main text'),
              f'tab:externalall{part + 1}', size='\\scriptsize')
    for old in ('table_provenance.tex',):
        (dest / old).unlink(missing_ok=True)
    print(f'Generated {len(macros)} macros and 18 tables.')


if __name__ == '__main__':
    main()

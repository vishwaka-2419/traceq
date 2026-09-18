#!/usr/bin/env python3
"""Recompute all summary statistics from the delivered realization-level CSV."""
from __future__ import annotations
import csv
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path
import numpy as np
from traceq.analysis import summarize, certified_violation_bound
ROOT = Path(__file__).resolve().parents[1]

def main() -> None:
    study = json.loads((ROOT / 'data/results/study.json').read_text())
    grouped = defaultdict(list)
    with gzip.open(ROOT / 'data/results/raw_experiments.csv.gz', 'rt') as stream:
        for row in csv.DictReader(stream):
            grouped[row['experiment']].append(row)
    assert not study['quick'], 'Published tables must not be built from smoke results.'
    assert set(grouped) == {r['id'] for r in study['results']}
    checks = 0
    for result in study['results']:
        rows = grouped[result['id']]
        times = [float(row['makespan']) for row in rows]
        recomputed = summarize(times, result['stats']['threshold'])
        for name, value in recomputed.items():
            assert math.isclose(value, result['stats'][name], rel_tol=1e-11, abs_tol=1e-9), (result['id'], name)
            checks += 1
        for i, row in enumerate(rows):
            assert int(row['run']) == i
            assert int(row['seed']) == result['seedbase'] + i
            b0 = int(row['B']) if row['prefill'] == 'True' else 0
            assert b0 + int(row['accepted']) == result['signature']['states'] + int(row['leftover'])
            assert math.isclose(float(row['makespan']) - float(row['stall']), result['t0'], abs_tol=1e-7)
        for stored, key in [('mean_residency', 'mean_residency'), ('mean_blocked_fraction', 'blocked_fraction'), ('mean_leftover', 'leftover')]:
            assert math.isclose(np.mean([float(row[key]) for row in rows]), result[stored], rel_tol=1e-11, abs_tol=1e-9)
        if 'validation' in result:
            v = result['validation']
            upper = certified_violation_bound(recomputed['violations'], len(rows), 1 - .05/v['family_size'])
            assert math.isclose(upper, v['violation_ucb_family'], rel_tol=1e-12)
    report = {
        'status': 'PASS',
        'batches': len(grouped),
        'raw_realizations': sum(map(len, grouped.values())),
        'summary_fields_recomputed': checks,
        'row_checks': ['realization/seed ordering', 'state conservation', 'runtime equals floor plus stall'],
        'validation_bounds_recomputed': sum('validation' in r for r in study['results']),
    }
    (ROOT / 'data/results/artifact_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()

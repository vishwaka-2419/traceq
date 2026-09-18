#!/usr/bin/env python3
"""Replay two predeclared stored realizations per batch through both simulators."""
from __future__ import annotations
import csv,gzip,json,time
from pathlib import Path
import numpy as np
from traceq.model import Config,simulate,simulate_reference,sample_latencies
ROOT=Path(__file__).resolve().parents[1]

def main():
    started=time.monotonic()
    study=json.loads((ROOT/'data/results/study.json').read_text())
    selected={r['id']:{0,r['stats']['n']-1} for r in study['results']}
    rows={}
    with gzip.open(ROOT/'data/results/raw_experiments.csv.gz','rt') as f:
        for row in csv.DictReader(f):
            if int(row['run']) in selected[row['experiment']]:
                rows[row['experiment'],int(row['run'])]=row
    base={mp:np.loadtxt(ROOT/f'data/traces/hubbard_6x4_{mp}.csv',delimiter=',',skiprows=1,dtype=int)[:,1] for mp in ['JW','TT']}
    checks=[];count=0
    for r in study['results']:
        rid=r['id'];m=base[r['mapping']]
        if rid.startswith('order_'):
            m=np.loadtxt(ROOT/f'data/traces/TT_{rid}.csv',delimiter=',',skiprows=1,dtype=int)[:,1]
        elif rid.startswith('shuffle_ensemble_'):
            m=np.random.default_rng(84000+int(rid.split('_')[-1])).permutation(m)
        m=np.tile(m,r['repetitions'])
        c=Config(**r['config'])
        for i in sorted(selected[rid]):
            raw=rows[rid,i];seed=r['seedbase']+i
            services=sample_latencies(c,int(m.sum())+c.buffer+4,seed)
            if r['mode']=='staggered':
                services[:,0]=c.mean_latency*(np.arange(c.producers)+1)/c.producers
            elif r['mode']=='common_slowdown':
                rng=np.random.default_rng(np.random.SeedSequence([seed,99999]))
                services*=.75 if rng.random()<.8 else 2.
            fast=simulate(m,c,services);ref=simulate_reference(m,c,services)
            for key in fast:
                np.testing.assert_allclose(fast[key],ref[key],rtol=1e-12,atol=1e-8,err_msg=f'{rid}/{i}/{key}')
                count+=1
            for key in ['makespan','stall','mean_residency','blocked_fraction','accepted','leftover','peak_inventory']:
                np.testing.assert_allclose(fast[key],float(raw[key]),rtol=1e-12,atol=1e-8,err_msg=f'raw {rid}/{i}/{key}')
            checks.append(dict(experiment=rid,run=i,seed=seed,status='PASS'))
    report=dict(status='PASS',batches=len(study['results']),replayed_realizations=len(checks),paired_fields_compared=count,
                selection='First and last stored realization in every batch; fixed deterministic selection.',runtime_seconds=time.monotonic()-started,checks=checks)
    (ROOT/'data/results/replay_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='checks'},indent=2))
if __name__=='__main__':main()

#!/usr/bin/env python3
"""Reproduce the complete numerical study. --quick is only a smoke run."""
from __future__ import annotations
import argparse, csv, gzip, json, math, platform, sys, time
from pathlib import Path
from dataclasses import asdict
import numpy as np
from traceq.model import Config, simulate, sample_latencies
from traceq.workloads import export_workload
from traceq.analysis import summarize, aggregate_signature, balanced_order, fluid_buffer_requirement, certified_violation_bound
ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--quick',action='store_true')
    parser.add_argument('--output-root',type=Path,help='Directory containing generated traces/ and results/')
    args=parser.parse_args();quick=args.quick
    data_root=args.output_root or (ROOT/'data/smoke' if quick else ROOT/'data')
    out=data_root/'results';out.mkdir(parents=True,exist_ok=True)
    trace_dir=data_root/'traces';trace_dir.mkdir(parents=True,exist_ok=True)
    ns=24 if quick else 512; nm=64 if quick else 4096; no=32 if quick else 1024
    ng=16 if quick else 128; nr=16 if quick else 256; nv=64 if quick else 2048
    started=time.time(); traces={}; meta={}; results=[]; byid={}
    for mapping in ['JW','TT']:
        traces[mapping],meta[mapping]=export_workload(trace_dir,mapping=mapping)
    fields=['experiment','run','seed','mapping','K','B','p','law','prefill','repetitions','makespan','stall','mean_residency','blocked_fraction','accepted','leftover','peak_inventory']
    raw=gzip.open(out/'raw_experiments.csv.gz','wt',newline='')
    writer=csv.DictWriter(raw,fieldnames=fields);writer.writeheader()

    def run(name,mapping,m,c,n,seedbase,repetitions=1,mode=None):
        vals=[]; ages=[]; blocks=[]; unused=[]
        for i in range(n):
            seed=seedbase+i;services=None
            if mode=='staggered':
                services=sample_latencies(c,int(m.sum())+c.buffer+4,seed)
                services[:,0]=c.mean_latency*(np.arange(c.producers)+1)/c.producers
            elif mode=='common_slowdown':
                services=sample_latencies(c,int(m.sum())+c.buffer+4,seed)
                rng=np.random.default_rng(np.random.SeedSequence([seed,99999]))
                services *= .75 if rng.random()<.8 else 2.
            x=simulate(m,c,services=services,seed=seed)
            vals.append(x['makespan']);ages.append(x['mean_residency']);blocks.append(x['blocked_fraction']);unused.append(x['leftover'])
            writer.writerow(dict(experiment=name,run=i,seed=seed,mapping=mapping,K=c.producers,B=c.buffer,p=c.acceptance,
                law=c.law if mode is None else mode,prefill=c.prefill,repetitions=repetitions,
                **{k:x[k] for k in ['makespan','stall','mean_residency','blocked_fraction','accepted','leftover','peak_inventory']}))
        t0=len(m)*c.step_time;trate=max(t0,int(m.sum())*c.mean_latency/c.producers)
        r=dict(id=name,mapping=mapping,config=asdict(c),mode=mode,repetitions=repetitions,t0=t0,trate=trate,
               signature=aggregate_signature(m),stats=summarize(vals,1.02*t0),mean_residency=float(np.mean(ages)),
               mean_blocked_fraction=float(np.mean(blocks)),mean_leftover=float(np.mean(unused)),
               rate_excess_pct=100*(np.mean(vals)/trate-1),floor_excess_pct=100*(np.mean(vals)/t0-1),
               fluid=fluid_buffer_requirement(m,c.producers/c.mean_latency,c.step_time),seedbase=seedbase)
        results.append(r);byid[name]=r
        return r

    for mapping,m in traces.items():
        ka=math.ceil(int(m.sum())*112.5/(len(m)*17));meta[mapping]['kavg']=ka
        r=run(f'baseline_{mapping}',mapping,m,Config(ka),nm,1000000)
        print('baseline',mapping,r['rate_excess_pct'],flush=True)
    for mapping,m in traces.items():
        for K in range(6,25): run(f'capacity_{mapping}_{K}',mapping,m,Config(K),ns,2000000)
        print('capacity',mapping,'done',round(time.time()-started,1),flush=True)
    m=traces['TT'];variants=dict(compiled=m,shuffled=np.random.default_rng(73129).permutation(m),interleaved=balanced_order(m),clustered=np.sort(m))
    for variant,mm in variants.items():
        np.savetxt(trace_dir/f'TT_order_{variant}.csv',np.c_[np.arange(len(mm)),mm],fmt='%d',delimiter=',',header='step,states',comments='')
        assert aggregate_signature(mm)==aggregate_signature(m)
        r=run(f'order_{variant}','TT',mm,Config(meta['TT']['kavg']),no,3000000)
        print('order',variant,r['rate_excess_pct'],flush=True)
    for j in range(4 if quick else 32):
        mm=np.random.default_rng(84000+j).permutation(m)
        run(f'shuffle_ensemble_{j}','TT',mm,Config(meta['TT']['kavg']),8 if quick else 64,3100000+j*1000)
    for mapping,m in traces.items():
        ka=meta[mapping]['kavg']
        for B in [4,8,12,16,24,32,64]: run(f'buffer_{mapping}_{B}',mapping,m,Config(ka,B),ns,4000000)
        for p in [.04,.08,.12,.2,.35,.5,.75,.9]:
            k=math.ceil(int(m.sum())*(9/p)/(len(m)*17))
            run(f'acceptance_{mapping}_{p}',mapping,m,Config(k,acceptance=p),ns,5000000)
        for law in ['deterministic','two_point','geometric']:
            run(f'law_{mapping}_{law}',mapping,m,Config(ka,law=law),2 if law=='deterministic' else ns,6000000)
        run(f'law_{mapping}_staggered',mapping,m,Config(ka,law='deterministic'),2,6000000,mode='staggered')
        run(f'law_{mapping}_common_slowdown',mapping,m,Config(ka),ns,6100000,mode='common_slowdown')
        for rep in [1,2,4,8]: run(f'repeated_{mapping}_{rep}',mapping,np.tile(m,rep),Config(ka),nr,7000000,repetitions=rep)
        run(f'cold_{mapping}',mapping,m,Config(ka,prefill=False),ns,7100000)
        print('sensitivity',mapping,'done',round(time.time()-started,1),flush=True)
    candidates=[]
    for mapping,m in traces.items():
        for B in [4,8,16,32]:
            screen=[]
            for K in range(6,25):
                r=byid[f'capacity_{mapping}_{K}'] if B==4 else run(f'grid_{mapping}_{K}_{B}',mapping,m,Config(K,B),ng,8000000)
                screen.append(r)
            good=[r for r in screen if r['stats']['violation_rate']<=.02]
            if not good: raise RuntimeError('no screened candidate; expand K grid')
            chosen=min(good,key=lambda r:r['config']['producers'])
            candidates.append(dict(mapping=mapping,K=chosen['config']['producers'],B=B,screen_id=chosen['id']))
        print('screen',mapping,'done',round(time.time()-started,1),flush=True)
    (out/'selected_candidates.json').write_text(json.dumps(candidates,indent=2)+'\n')
    M=len(candidates)
    for i,cand in enumerate(candidates):
        mapping=cand['mapping'];K=cand['K'];B=cand['B']
        r=run(f'validation_{mapping}_{K}_{B}',mapping,traces[mapping],Config(K,B),nv,9000000+i*10000)
        x=r['stats']['violations']; u=certified_violation_bound(x,nv,1-.05/M)
        r['validation']=dict(family_size=M,confidence_family=.95,violation_ucb_family=u,certified=u<=.05,screen_id=cand['screen_id'])
        print('validation',mapping,K,B,r['validation'],flush=True)
    exact=[]
    for w in [2,3,4,8]:
        for rep in [1,2,4,8,16,64]:
            A=np.tile([w]+[0]*(w-1),2*rep);B=np.tile([w,w]+[0]*(2*w-2),rep)
            cfg=Config(1,w,1,1,1,True,'deterministic')
            ta=simulate(A,cfg)['makespan'];tb=simulate(B,cfg)['makespan']
            assert ta==2*w*rep and tb==(3*w-2)*rep+1
            exact.append(dict(w=w,repeats=rep,A=ta,B=tb,excess_pct=100*(tb/ta-1),minimax_relative_error=(tb-ta)/(tb+ta)))
    raw.close()
    import scipy,numba,matplotlib
    payload=dict(study_version='0.1.0',date='2026-09-17',quick=quick,elapsed_seconds=time.time()-started,
                 workload_metadata=meta,results=results,exact=exact,
                 sample_sizes=dict(main=nm,sweep=ns,order=no,grid=ng,repeated=nr,validation=nv),
                 environment=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,numba=numba.__version__,matplotlib=matplotlib.__version__,platform=platform.platform()))
    (out/'study.json').write_text(json.dumps(payload,indent=2)+'\n')
    print('DONE',len(results),'design batches; seconds',round(time.time()-started,1),flush=True)
if __name__=='__main__': main()

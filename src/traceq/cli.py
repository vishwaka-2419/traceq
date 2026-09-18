"""Run an externally supplied demand trace: python -m traceq.cli --help."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from .model import Config,simulate
from .analysis import summarize,fluid_buffer_requirement
from .theory import deterministic_runtime,fluid_runtime,histogram_envelope

def closed_form(m,c):
    """Deterministic-limit quantities of the paper; no random numbers involved."""
    rho,mu,K,B,tau=c.producers/c.mean_latency,c.mean_latency,c.producers,c.buffer,c.step_time
    out=dict(deterministic_runtime=deterministic_runtime(m,K,B,mu,tau,prefill=c.prefill),
             note='Exact for constant equal latencies; an estimate of the mean otherwise. No tail information.')
    if c.prefill:
        out['fluid_bracket']=[fluid_runtime(m,rho,B+K,tau),fluid_runtime(m,rho,B+1,tau)+mu]
        if B+1>=rho*tau:
            lo=histogram_envelope(m,rho,B+K,tau)['lower'];hi=histogram_envelope(m,rho,B+1,tau)['upper']+mu
            out['histogram_envelope']=[lo,hi]
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('trace',type=Path,help='CSV with a states column and optional step column')
    p.add_argument('--producers','-K',type=int,required=True)
    p.add_argument('--buffer','-B',type=int,default=4)
    p.add_argument('--step-time',type=float,default=17.)
    p.add_argument('--attempt-time',type=float,default=9.)
    p.add_argument('--acceptance',type=float,default=.08)
    p.add_argument('--law',choices=['geometric','deterministic','two_point'],default='geometric')
    p.add_argument('--cold',action='store_true',help='start with zero stored states')
    p.add_argument('--trials',type=int,default=512)
    p.add_argument('--seed',type=int,default=420000)
    p.add_argument('--deadline-factor',type=float,default=1.02)
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    if a.trials<2:p.error('--trials must be at least two')
    try:
        data=np.genfromtxt(a.trace,delimiter=',',names=True)
        if data.dtype.names is None or 'states' not in data.dtype.names:
            raise ValueError('CSV must have a states column')
        m=np.atleast_1d(data['states'])
        c=Config(a.producers,a.buffer,a.step_time,a.attempt_time,a.acceptance,not a.cold,a.law)
        runs=[simulate(m,c,seed=a.seed+i) for i in range(a.trials)]
        t0=len(m)*c.step_time
        out=dict(trace=str(a.trace),supply_unconstrained_time=t0,
                 runtime=summarize([r['makespan'] for r in runs],a.deadline_factor*t0),
                 mean_residency=float(np.mean([r['mean_residency'] for r in runs])),
                 mean_blocked_fraction=float(np.mean([r['blocked_fraction'] for r in runs])),
                 fluid=fluid_buffer_requirement(m,c.producers/c.mean_latency,c.step_time),
                 closed_form=closed_form(m,c),
                 warning='The binomial bound is model-conditional; use independent validation after design selection.')
        text=json.dumps(out,indent=2)+'\n'
        if a.output:a.output.write_text(text)
        else:print(text,end='')
    except (ValueError,OSError) as e:
        p.error(str(e))
if __name__=='__main__':main()

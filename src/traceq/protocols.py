"""Optional stage-resolved supply adapter; parameters require physical calibration."""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np

@dataclass(frozen=True)
class Stage:
    duration: float
    acceptance: float
    name: str = ''
    def __post_init__(self):
        if not math.isfinite(self.duration) or self.duration<=0:
            raise ValueError('duration must be positive and finite')
        if not math.isfinite(self.acceptance) or not 0<self.acceptance<=1:
            raise ValueError('acceptance must be in (0,1]')


def staged_moments(stages, reset_time=0.0):
    """Exact moments for independent stage tests and restart after any failure.

    Each reached stage spends its entire declared duration before acceptance
    is known. Failure returns to stage 1 after reset_time. Success of all stages
    creates one usable output. No state fidelity is inferred from acceptance.
    """
    stages=tuple(stages)
    if not stages or not math.isfinite(reset_time) or reset_time<0:
        raise ValueError('provide stages and a nonnegative finite reset time')
    reach=1.;prefix=0.;ed=ed2=0.
    for s in stages:
        prefix+=s.duration
        failure=reach*(1-s.acceptance)
        ed+=failure*prefix;ed2+=failure*prefix**2
        reach*=s.acceptance
    p=reach
    if p<=0: raise ValueError('acceptance product underflows')
    ed+=p*prefix;ed2+=p*prefix**2
    ed_fail=ed-p*prefix
    mean=(ed+(1-p)*reset_time)/p
    second=(ed2+2*reset_time*ed_fail+reset_time**2*(1-p)
            +2*(ed_fail+reset_time*(1-p))*mean)/p
    return dict(acceptance=p,successful_attempt_time=prefix,mean=mean,
                variance=max(0.,second-mean**2),mean_attempt_time=ed)


def sample_staged(stages, number, seed, reset_time=0.0):
    stages=tuple(stages);staged_moments(stages,reset_time)
    if number<1: raise ValueError('number must be positive')
    rng=np.random.default_rng(seed);out=np.zeros(number)
    for i in range(number):
        while True:
            success=True
            for stage in stages:
                out[i]+=stage.duration
                if rng.random()>=stage.acceptance:
                    out[i]+=reset_time;success=False;break
            if success: break
    return out


def sample_staged_fast(stages, shape, seed, reset_time=0.0):
    """Vectorised sampler with the law of ``sample_staged``.

    The number of failed attempts before the first success is geometric. Each failed
    attempt ends at stage i with probability reach_i*(1-p_i)/(1-P) and then costs the
    time spent up to and including that stage plus the reset time.
    """
    stages=tuple(stages);mom=staged_moments(stages,reset_time)
    rng=np.random.default_rng(seed)
    reach=1.;prefix=0.;weights=[];costs=[]
    for s in stages:
        prefix+=s.duration
        weights.append(reach*(1-s.acceptance));costs.append(prefix+reset_time)
        reach*=s.acceptance
    out=np.full(shape,prefix,dtype=float)
    if reach<1:
        failures=rng.geometric(reach,size=shape)-1
        q=np.array(weights)/(1-reach)
        counts=rng.multinomial(failures.ravel(),q)
        out+=(counts@np.array(costs)).reshape(shape)
    return out

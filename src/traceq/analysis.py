"""Order-sensitive fluid diagnostic and finite-sample reporting."""
from __future__ import annotations
import math
import numpy as np
from scipy.stats import beta, t as student_t
from .model import validate_trace


def fluid_buffer_requirement(demand, rate: float, step_time: float = 1.0):
    """Exact minimum real-valued storage for the constant-rate fluid model.

    Atomic demands at 0,tau,2*tau,...; full initial storage; supply between
    demands at fixed rate, curtailed at capacity. This is NOT a stochastic
    lower bound evaluated at the mean rate, nor is it the external B needed
    by a blocking-after-service producer network.
    """
    m = validate_trace(demand)
    if rate < 0 or not math.isfinite(rate) or step_time <= 0 or not math.isfinite(step_time):
        raise ValueError('invalid rate or step_time')
    deficit = maximum = 0.0
    best = (0,0)
    start = 0
    for s, a in enumerate(m):
        required = deficit+float(a)
        if required > maximum:
            maximum = required
            best = (start, s)
        deficit = max(0., required-rate*step_time)
        if deficit == 0.0:
            start = s+1
    return dict(capacity=maximum, start=best[0], stop=best[1])


def certified_violation_bound(failures: int, trials: int, confidence=0.95):
    """One-sided exact Clopper-Pearson upper bound on deadline violation.

    The certification is conditional on the declared simulation model,
    independent validation draws, and a candidate fixed before validation.
    """
    if (isinstance(trials, bool) or not isinstance(trials, int) or trials < 1
            or isinstance(failures, bool) or not isinstance(failures, int)
            or not 0 <= failures <= trials):
        raise ValueError('require 0 <= failures <= positive trials')
    if not 0 < confidence < 1:
        raise ValueError('confidence must be in (0,1)')
    if failures == trials:
        return 1.0
    return float(beta.ppf(confidence, failures+1, trials-failures))


def summarize(values, threshold=None):
    a = np.asarray(values, dtype=float)
    if a.ndim != 1 or len(a) < 2 or not np.all(np.isfinite(a)):
        raise ValueError('at least two finite observations are needed')
    mean = float(a.mean())
    std = float(a.std(ddof=1))
    half = float(student_t.ppf(.975,len(a)-1)*std/np.sqrt(len(a)))
    out = dict(n=len(a), mean=mean, sd=std, mean_ci_low=mean-half, mean_ci_high=mean+half,
               cv=std/mean if mean else 0.,
               q95=float(np.quantile(a,.95,method='higher')),
               q99=float(np.quantile(a,.99,method='higher')),
               minimum=float(a.min()), maximum=float(a.max()))
    if threshold is not None:
        x = int(np.count_nonzero(a>threshold))
        out.update(violations=x, threshold=float(threshold),
                   violation_rate=x/len(a), violation_ucb95=certified_violation_bound(x,len(a)))
    return out


def balanced_order(demand):
    """Histogram-preserving deterministic interleaving by evenly spaced ranks.

    Within each demand category with c occurrences, use positions (j+.5)/c.
    Merge categories by position, breaking ties by increasing demand. These
    artificial permutations need NOT be executable schedules of a circuit.
    """
    m = validate_trace(demand)
    vals, counts = np.unique(m, return_counts=True)
    merged = []
    for v,c in zip(vals,counts):
        merged.extend(((j+.5)/int(c),int(v),j) for j in range(int(c)))
    return np.array([v for _,v,_ in sorted(merged)],dtype=np.int64)


def aggregate_signature(demand):
    m = validate_trace(demand)
    vals,counts = np.unique(m,return_counts=True)
    return dict(states=int(m.sum()), steps=len(m), peak=int(m.max()), mean=float(m.mean()),
                histogram={str(int(v)):int(c) for v,c in zip(vals,counts)})

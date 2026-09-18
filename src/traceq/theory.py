"""Order-sensitive closed forms for the deterministic limit of the queue.

Three objects are provided.

1. ``deterministic_runtime``: the exact makespan of the blocking-after-service
   queue when every producer has the same constant latency. The dynamics are
   (max,+)-linear, so no event queue is needed; cost is O(S + N_T).
2. ``fluid_runtime``: the makespan of a constant-rate reservoir with finite
   capacity, equal to a maximum over families of disjoint overload intervals.
   ``fluid_overload_intervals`` returns a maximising family.
3. ``histogram_envelope``: the largest and smallest fluid makespan compatible
   with a demand histogram, from the rearrangement theorem of the paper.

Nothing here samples random numbers. None of it predicts tails, residency
distributions or deadline probabilities; use the simulator for those.
"""
from __future__ import annotations
import math
import numpy as np
from numba import njit
from .model import validate_trace

__all__ = [
    "deterministic_runtime", "deterministic_interval_formula",
    "fluid_runtime", "fluid_interval_formula", "fluid_overload_intervals",
    "histogram_envelope", "cluster_family", "cluster_family_runtimes",
    "minimax_floor",
]


# --------------------------------------------------------------------------
# exact deterministic runtime: (max,+) recursion
# --------------------------------------------------------------------------
@njit(cache=True)
def _det_kernel(m, K, B, mu, tau, prefill, staggered):
    S = len(m)
    N = 0
    for a in m:
        N += a
    B0 = B if prefill else 0
    size = N + B + K + 3
    entry = np.zeros(size)              # e_i : time state i enters the buffer
    taken = np.full(size, -np.inf)      # time state i is consumed
    starts = np.empty(S)
    D = 0
    nxt = B0 + 1
    ready = 0.0
    for s in range(S):
        a = m[s]
        D += a
        while nxt <= D:
            i = nxt
            n = i - B0
            if n <= K:
                done = (n * mu / K) if staggered else mu
            else:
                done = entry[i - K] + mu
            freed = taken[i - B] if i - B >= 1 else -np.inf
            entry[i] = done if done > freed else freed
            nxt += 1
        start = ready
        if a > 0 and D > B0 and entry[D] > start:
            start = entry[D]
        for i in range(D - a + 1, D + 1):
            taken[i] = start
        starts[s] = start
        ready = start + tau
    return ready, starts


def deterministic_runtime(demand, producers, buffer, latency, step_time=1.0,
                          prefill=True, staggered=False, return_starts=False):
    """Exact makespan for K producers with one constant latency.

    Index accepted states in the order they enter the buffer (prefilled states
    first). With e_i the entry time of state i and sigma_s the start of step s,

        sigma_s = max(sigma_{s-1} + tau, e_{D_s}),
        e_i     = max(a_i, sigma_{s(i-B)}),
        a_i     = e_{i-K} + mu            (first K completions: mu, or
                                           j*mu/K when ``staggered``),

    where D_s is cumulative demand and s(i) the step consuming state i.
    """
    m = validate_trace(demand)
    K, B = int(producers), int(buffer)
    if K < 1 or B < 1 or latency <= 0 or step_time <= 0:
        raise ValueError("producers, buffer, latency and step_time must be positive")
    if int(m.max()) > B:
        raise ValueError("atomic demand exceeds external buffer capacity")
    total, starts = _det_kernel(m, K, B, float(latency), float(step_time),
                                bool(prefill), bool(staggered))
    return (float(total), starts) if return_starts else float(total)


def deterministic_interval_formula(demand, producers, buffer, latency, step_time=1.0):
    """Closed form of Theorem 2 by O(S^2) dynamic programming (small traces).

    T = S*tau + max over families of disjoint intervals of the summed gains
    g(I) = mu*floor((D(I)-B-1)/K) - (|I|-1)*tau, with an extra +mu for an
    interval that contains step 1. Aligned producers, prefilled buffer.
    """
    m = validate_trace(demand)
    K, B, mu, tau = int(producers), int(buffer), float(latency), float(step_time)
    S = len(m)
    pre = np.concatenate([[0], np.cumsum(m)])
    best = np.zeros(S + 1)
    for j in range(1, S + 1):
        b = best[j - 1]
        for i in range(1, j + 1):
            D = int(pre[j] - pre[i - 1])
            g = mu * math.floor((D - B - 1) / K) - (j - i) * tau
            if i == 1:
                b = max(b, g + mu)
            else:
                b = max(b, best[i - 1] + g)
        best[j] = b
    return S * tau + float(best[S])


# --------------------------------------------------------------------------
# fluid reservoir
# --------------------------------------------------------------------------
@njit(cache=True)
def _fluid_kernel(m, rho, C, tau, deficit0):
    S = len(m)
    starts = np.empty(S)
    x = deficit0
    t = 0.0
    for s in range(S):
        over = x + m[s] - C
        if over > 0.0:
            t += over / rho
            x = C
        else:
            x = x + m[s]
        starts[s] = t
        t += tau
        x = x - rho * tau
        if x < 0.0:
            x = 0.0
    return t, starts


def _check_fluid(m, rate, capacity, step_time):
    if not (rate > 0 and math.isfinite(rate) and step_time > 0 and math.isfinite(step_time)):
        raise ValueError("rate and step_time must be finite and positive")
    if capacity < float(m.max()):
        raise ValueError("capacity must be at least the largest atomic demand")


def fluid_runtime(demand, rate, capacity, step_time=1.0, initial_deficit=0.0,
                  return_starts=False):
    """Makespan of a reservoir refilled at ``rate`` and capped at ``capacity``.

    Two-sided Lindley recursion on the missing inventory x_s:
        stall_s = (x_s + m_s - C)_+ / rho,
        x_{s+1} = max(0, min(x_s + m_s, C) - rho*tau).
    """
    m = validate_trace(demand)
    _check_fluid(m, rate, capacity, step_time)
    total, starts = _fluid_kernel(m, float(rate), float(capacity), float(step_time),
                                  float(initial_deficit))
    return (float(total), starts) if return_starts else float(total)


def fluid_interval_formula(demand, rate, capacity, step_time=1.0):
    """S*tau + (1/rho) max over disjoint interval families of sum(E(I)-C); O(S^2)."""
    m = validate_trace(demand)
    _check_fluid(m, rate, capacity, step_time)
    S = len(m)
    pre = np.concatenate([[0], np.cumsum(m)])
    best = np.zeros(S + 1)
    for j in range(1, S + 1):
        b = best[j - 1]
        for i in range(1, j + 1):
            g = (pre[j] - pre[i - 1] - rate * step_time * (j - i) - capacity) / rate
            b = max(b, best[i - 1] + g)
        best[j] = b
    return S * step_time + float(best[S])


def fluid_overload_intervals(demand, rate, capacity, step_time=1.0):
    """A maximising family of disjoint overload intervals, in O(S).

    Returns a list of dicts with 0-based inclusive ``start``/``stop`` and the
    stall ``gain`` (time units) each interval contributes. The gains sum to
    fluid_runtime - S*tau.
    """
    m = validate_trace(demand)
    _check_fluid(m, rate, capacity, step_time)
    rho, C, tau = float(rate), float(capacity), float(step_time)
    S = len(m)
    sig = np.empty(S)
    F = np.empty(S)
    sig_from_F = np.zeros(S, dtype=bool)
    F_from_F = np.zeros(S, dtype=bool)
    eps = 1e-12
    prev_sig_end, prevF = 0.0, 0.0
    for s in range(S):
        viaF = prevF - (C - m[s]) / rho
        if viaF > prev_sig_end + eps:
            sig[s] = viaF
            sig_from_F[s] = True
        else:
            sig[s] = prev_sig_end
        if prevF > sig[s] + eps:
            F[s] = prevF + m[s] / rho
            F_from_F[s] = True
        else:
            F[s] = sig[s] + m[s] / rho
        prev_sig_end, prevF = sig[s] + tau, F[s]
    out = []
    s = S - 1
    pre = np.concatenate([[0], np.cumsum(m)])
    while s >= 0:
        if sig_from_F[s]:
            j = s - 1
            while j >= 0 and F_from_F[j]:
                j -= 1
            j = max(j, 0)
            # sigma_j may itself be reached from F (only when m_j equals C);
            # the two intervals then share step j and are reported merged.
            while j > 0 and sig_from_F[j]:
                j2 = j - 1
                while j2 >= 0 and F_from_F[j2]:
                    j2 -= 1
                j = max(j2, 0)
            D = pre[s + 1] - pre[j]
            out.append(dict(start=int(j), stop=int(s),
                            gain=float((D - C) / rho - (s - j) * tau)))
            s = j - 1 if j < s else s - 1
        else:
            s -= 1
    out.reverse()
    return out


# --------------------------------------------------------------------------
# what the histogram alone allows
# --------------------------------------------------------------------------
def histogram_envelope(demand, rate, capacity, step_time=1.0):
    """Largest and smallest fluid makespan over all orderings of a multiset.

    upper = S*tau + [sum_s (m_s - rho*tau)_+ - (C - rho*tau)]_+ / rho, attained
    by the sorted arrangement; lower = max(S*tau, tau + (N_T - C)/rho).
    Requires capacity >= rate*step_time (storage holds one step of supply).
    """
    m = validate_trace(demand).astype(float)
    _check_fluid(m, rate, capacity, step_time)
    rt = rate * step_time
    if capacity < rt:
        raise ValueError("envelope requires capacity >= rate*step_time")
    S = len(m)
    over = float(np.clip(m - rt, 0.0, None).sum())
    upper = S * step_time + max(0.0, over - (capacity - rt)) / rate
    lower = max(S * step_time, step_time + (float(m.sum()) - capacity) / rate)
    return dict(lower=float(lower), upper=float(upper), overload_mass=over)


# --------------------------------------------------------------------------
# the exact separation family of Theorem 1
# --------------------------------------------------------------------------
def cluster_family(width, cluster, repeats):
    """Equal-histogram pair: (w,0^{w-1})^{c r} and (w^c,0^{c(w-1)})^r."""
    w, c, r = int(width), int(cluster), int(repeats)
    if w < 2 or c < 2 or r < 1:
        raise ValueError("need width >= 2, cluster >= 2, repeats >= 1")
    spread = np.tile(np.array([w] + [0] * (w - 1), dtype=np.int64), c * r)
    clustered = np.tile(np.array([w] * c + [0] * (c * (w - 1)), dtype=np.int64), r)
    return spread, clustered


def cluster_family_runtimes(width, cluster, repeats):
    """Exact runtimes for K=1, B=w, tau=mu=1: c*w*r and ((2c-1)w-c)*r+1."""
    w, c, r = int(width), int(cluster), int(repeats)
    return c * w * r, ((2 * c - 1) * w - c) * r + 1


def minimax_floor(t_small, t_large):
    """Least worst-case relative error of one scalar guess for two runtimes."""
    if not 0 < t_small <= t_large:
        raise ValueError("need 0 < t_small <= t_large")
    return (t_large - t_small) / (t_large + t_small)

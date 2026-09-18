"""Explicit blocking-after-service queue with atomic, ordered demands.

Units are arbitrary consistent time units (code cycles in the supplied study).
A successful state that cannot enter the external FIFO stays at its producer;
that producer does not restart until the state is transferred. The producer's
holding space is NOT part of B. At most B+K accepted states can therefore exist.
Completion events precede demand events at equal times, then tied completions
are processed by ascending producer index. There is no delivery delay, expiry,
correlation, physical-noise simulation, or dynamic rescheduling in this model.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from collections import deque
from typing import Literal
import math
import heapq
import numpy as np
from numba import njit

@dataclass(frozen=True)
class Config:
    producers: int
    buffer: int = 4
    step_time: float = 17.0
    attempt_time: float = 9.0
    acceptance: float = 0.08
    prefill: bool = True
    law: Literal['geometric', 'deterministic', 'two_point'] = 'geometric'

    def __post_init__(self):
        if isinstance(self.producers, bool) or not isinstance(self.producers, int) or self.producers < 1:
            raise ValueError('producers must be a positive integer')
        if isinstance(self.buffer, bool) or not isinstance(self.buffer, int) or self.buffer < 1:
            raise ValueError('buffer must be a positive integer')
        if not math.isfinite(self.step_time) or self.step_time <= 0:
            raise ValueError('step_time must be finite and positive')
        if not math.isfinite(self.attempt_time) or self.attempt_time <= 0:
            raise ValueError('attempt_time must be finite and positive')
        if not math.isfinite(self.acceptance) or not 0 < self.acceptance <= 1:
            raise ValueError('acceptance must be in (0,1]')
        if not isinstance(self.prefill, bool):
            raise ValueError('prefill must be a boolean')
        if self.law not in ('geometric', 'deterministic', 'two_point'):
            raise ValueError('unknown service law')

    @property
    def mean_latency(self) -> float:
        return self.attempt_time / self.acceptance


def validate_trace(demand):
    a = np.asarray(demand)
    if a.ndim != 1 or len(a) == 0:
        raise ValueError('demand must be a nonempty one-dimensional sequence')
    if not np.issubdtype(a.dtype, np.number) or not np.all(np.isfinite(a)):
        raise ValueError('demand must contain finite numbers')
    if np.any(a < 0) or np.any(a != np.floor(a)):
        raise ValueError('demands must be nonnegative integers')
    return np.ascontiguousarray(a, dtype=np.int64)


def sample_latencies(config: Config, number: int, seed: int):
    """Return K independent service streams (one stream per producer).

    Geometric support is {1,2,...}; two_point has equal masses at a and
    2*mu-a, and shares the same mean as the other two laws. This is a
    distributional ablation, not an experimentally calibrated protocol.
    """
    if number < 1:
        raise ValueError('number must be positive')
    out = np.empty((config.producers, number), dtype=np.float64)
    for j in range(config.producers):
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), j]))
        if config.law == 'geometric':
            out[j] = config.attempt_time * rng.geometric(config.acceptance, size=number)
        elif config.law == 'deterministic':
            out[j].fill(config.mean_latency)
        else:
            out[j] = np.where(rng.random(number) < 0.5,
                              config.attempt_time,
                              2 * config.mean_latency - config.attempt_time)
    return out


def validate_services(services, config):
    a = np.ascontiguousarray(services, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != config.producers or a.shape[1] < 1:
        raise ValueError('services must have shape (producers, positive_number)')
    if not np.all(np.isfinite(a)) or np.any(a <= 0):
        raise ValueError('all service latencies must be finite and positive')
    return a


# Independent, readability-first heap/deque implementation, used to audit the
# accelerated state machine. Each supplied stream is indexed by producer.
def simulate_reference(demand, config: Config, services=None, seed=0):
    m = validate_trace(demand)
    if int(m.max()) > config.buffer:
        raise ValueError('atomic demand exceeds external buffer capacity')
    if services is None:
        services = sample_latencies(config, int(m.sum()) + config.buffer + 4, seed)
    services = validate_services(services, config)
    K, B = config.producers, config.buffer
    index = [1] * K
    events = [(services[j, 0], j) for j in range(K)]
    heapq.heapify(events)
    buffer = deque([0.0] * (B if config.prefill else 0))
    held = []  # heap of (completion time, producer), global completion FIFO
    time = 0.0
    accepted = consumed = 0
    blocked = residency = inventory_integral = 0.0
    starts = []
    ages = []
    peak = len(buffer)

    def advance(t):
        nonlocal time, inventory_integral
        if t < time:
            raise AssertionError('time reversal')
        inventory_integral += (len(buffer) + len(held)) * (t - time)
        time = float(t)

    def restart(j):
        if index[j] >= services.shape[1]:
            raise ValueError('service stream exhausted')
        heapq.heappush(events, (time + services[j, index[j]], j))
        index[j] += 1

    def completion():
        nonlocal accepted, peak
        t, j = heapq.heappop(events)
        advance(t)
        accepted += 1
        if len(buffer) < B:
            buffer.append(time)
            restart(j)
        else:
            heapq.heappush(held, (time, j))
        peak = max(peak, len(buffer) + len(held))

    def fill_from_held():
        nonlocal blocked
        while len(buffer) < B and held:
            born, j = heapq.heappop(held)
            blocked += time - born
            buffer.append(born)
            restart(j)

    target = 0.0
    for amount in m:
        while events and events[0][0] <= target:
            completion()
        advance(target)
        while len(buffer) < amount:
            if not events:
                raise AssertionError('unexpected deadlock')
            target = float(events[0][0])
            # Process every tied completion before the demand.
            while events and events[0][0] <= target:
                completion()
        starts.append(time)
        for _ in range(int(amount)):
            age = time - buffer.popleft()
            residency += age
            ages.append(age)
            consumed += 1
        fill_from_held()
        target = time + config.step_time
    while events and events[0][0] <= target:
        completion()
    advance(target)
    blocked += sum(time - born for born, _ in held)
    leftover_age = sum(time - born for born in buffer) + sum(time - born for born, _ in held)
    initial = B if config.prefill else 0
    assert initial + accepted == consumed + len(buffer) + len(held)
    assert abs(inventory_integral - residency - leftover_age) < 1e-6 * max(1, inventory_integral)
    return dict(makespan=time, stall=time-len(m)*config.step_time,
                residency_sum=residency, mean_residency=residency/max(1,consumed),
                blocked_patch_time=blocked, blocked_fraction=blocked/(K*time),
                accepted=accepted, consumed=consumed, leftover=len(buffer)+len(held),
                external_leftover=len(buffer), held_leftover=len(held), peak_inventory=peak,
                inventory_integral=inventory_integral, leftover_age=leftover_age,
                starts=np.array(starts), ages=np.array(ages))


@njit(cache=True)
def _kernel(m, K, B, tau, initial, services):
    """Event scan implementation, deliberately independent of heap reference."""
    # no more than N+B+K accepted states: no loss and finite inventory.
    N = 0
    for a in m:
        N += a
    births = np.empty(N + B + K + 2, np.float64)
    for i in range(initial):
        births[i] = 0.0
    head, tail = 0, initial
    due = services[:, 0].copy()
    pos = np.ones(K, np.int64)
    hold = np.full(K, -1.0)
    now = 0.0
    target = 0.0
    s = 0
    accepted = 0
    residency = 0.0
    blocked = 0.0
    inventory_integral = 0.0
    peak = initial
    starts = np.zeros(len(m), np.float64)
    held_count = 0
    # A demand happens only after all completions due <= target have run.
    # If stock is insufficient at target, move target to next completion.
    while True:
        j = 0
        for q in range(1, K):
            if due[q] < due[j]:
                j = q
        next_due = due[j]
        if next_due <= target:
            inventory_integral += (tail-head+held_count) * (next_due-now)
            now = next_due
            accepted += 1
            if tail-head < B:
                births[tail] = now
                tail += 1
                if pos[j] >= services.shape[1]:
                    raise ValueError('service stream exhausted')
                due[j] = now + services[j, pos[j]]
                pos[j] += 1
            else:
                hold[j] = now
                due[j] = np.inf
                held_count += 1
            if tail-head+held_count > peak:
                peak = tail-head+held_count
            continue
        inventory_integral += (tail-head+held_count) * (target-now)
        now = target
        if s == len(m):
            break
        if tail-head < m[s]:
            if not np.isfinite(next_due):
                raise ValueError('unexpected deadlock')
            target = next_due
            continue
        starts[s] = now
        for _ in range(m[s]):
            residency += now-births[head]
            head += 1
        # Release oldest held successes first; index is tie-breaker.
        while tail-head < B and held_count > 0:
            oldest = -1
            for q in range(K):
                if hold[q] >= 0 and (oldest < 0 or hold[q] < hold[oldest]):
                    oldest = q
            births[tail] = hold[oldest]
            tail += 1
            blocked += now-hold[oldest]
            hold[oldest] = -1.0
            held_count -= 1
            if pos[oldest] >= services.shape[1]:
                raise ValueError('service stream exhausted')
            due[oldest] = now+services[oldest, pos[oldest]]
            pos[oldest] += 1
        s += 1
        target = now+tau
    leftover_age = 0.0
    for q in range(head, tail):
        leftover_age += now-births[q]
    for q in range(K):
        if hold[q] >= 0:
            blocked += now-hold[q]
            leftover_age += now-hold[q]
    return (now, residency, blocked, accepted, tail-head, held_count,
            peak, inventory_integral, leftover_age, starts)


def simulate(demand, config: Config, services=None, seed=0):
    m = validate_trace(demand)
    if int(m.max()) > config.buffer:
        raise ValueError('atomic demand exceeds external buffer capacity')
    if services is None:
        services = sample_latencies(config, int(m.sum()) + config.buffer + 4, seed)
    services = validate_services(services, config)
    initial = config.buffer if config.prefill else 0
    t, rs, bt, ac, eb, eh, peak, integ, la, starts = _kernel(
        m, config.producers, config.buffer, config.step_time, initial, services)
    n = int(m.sum())
    if initial+ac != n+eb+eh:
        raise AssertionError('state conservation failure')
    if abs(integ-rs-la) > 1e-6 * max(1,integ):
        raise AssertionError('residency integral mismatch')
    return dict(makespan=float(t), stall=float(t-len(m)*config.step_time),
                residency_sum=float(rs), mean_residency=float(rs/max(1,n)),
                blocked_patch_time=float(bt), blocked_fraction=float(bt/(config.producers*t)),
                accepted=int(ac), consumed=n, leftover=int(eb+eh), external_leftover=int(eb),
                held_leftover=int(eh), peak_inventory=int(peak), inventory_integral=float(integ),
                leftover_age=float(la), starts=starts)

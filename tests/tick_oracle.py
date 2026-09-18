"""Small integer-clock oracle, independent of both event-based implementations.

Only for finite, positive integer service streams and integer step durations.
Each clock tick integrates inventory/blocked time, advances running services by
one, records all completions, and then considers the atomic consumer.
"""
from __future__ import annotations
import numpy as np


def tick_oracle(demand, config, services):
    m = list(map(int, demand))
    streams = np.asarray(services, dtype=int)
    assert np.array_equal(streams, services) and np.all(streams > 0)
    K, B, tau = config.producers, config.buffer, int(config.step_time)
    assert tau == config.step_time
    stock = [0] * (B if config.prefill else 0)
    held = {}  # producer -> acceptance tick
    remaining = list(map(int, streams[:, 0]))
    positions = [1] * K
    initial = len(stock)
    accepted = consumed = 0
    peak = initial
    inventory_integral = residency_sum = blocked_patch_time = 0
    starts = []
    t = ready = s = 0

    def restart(j):
        remaining[j] = int(streams[j, positions[j]])
        positions[j] += 1

    while t < 100000:
        if t > 0:
            inventory_integral += len(stock) + len(held)
            blocked_patch_time += len(held)
            finished = []
            for j in range(K):
                if j not in held:
                    remaining[j] -= 1
                    if remaining[j] == 0:
                        finished.append(j)
            for j in finished:  # ascending producer index resolves ties
                accepted += 1
                if len(stock) < B:
                    stock.append(t)
                    restart(j)
                else:
                    held[j] = t
            peak = max(peak, len(stock) + len(held))
        assert initial + accepted == consumed + len(stock) + len(held)
        assert 0 <= len(stock) <= B and len(held) <= K
        if s == len(m):
            if t == ready:
                break
        elif t >= ready and len(stock) >= m[s]:
            starts.append(t)
            for _ in range(m[s]):
                residency_sum += t - stock.pop(0)
                consumed += 1
            for j in sorted(held, key=lambda j: (held[j], j)):
                if len(stock) == B:
                    break
                stock.append(held.pop(j))
                restart(j)
            s += 1
            ready = t + tau
        assert initial + accepted == consumed + len(stock) + len(held)
        t += 1
    else:
        raise RuntimeError('clock oracle exceeded its small-case safety limit')
    leftover_age = sum(t - b for b in stock) + sum(t - b for b in held.values())
    assert inventory_integral == residency_sum + leftover_age
    return dict(makespan=t, stall=t-len(m)*tau, residency_sum=residency_sum,
                mean_residency=residency_sum/max(1,consumed),
                blocked_patch_time=blocked_patch_time, blocked_fraction=blocked_patch_time/(K*t),
                accepted=accepted, consumed=consumed, leftover=len(stock)+len(held),
                external_leftover=len(stock), held_leftover=len(held), peak_inventory=peak,
                inventory_integral=inventory_integral, leftover_age=leftover_age,
                starts=np.asarray(starts))

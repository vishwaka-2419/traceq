"""Integer-clock reference for alternative holding and injection policies.

Used only to test the policy-independent bound of the paper on small cases.
Every producer has the same integer latency. Policies:

    holding   'hold'     blocking after service (the declared model)
              'reserve'  a producer starts only with a reserved buffer slot
              'discard'  an output that finds the buffer full is dropped
    injection 'atomic'   a step takes all m_s states at once
              'incremental' the waiting consumer takes states as they arrive
"""
from __future__ import annotations
import numpy as np

HOLDING = ("hold", "reserve", "discard")
INJECTION = ("atomic", "incremental")


def inventory_cap(demand, producers, buffer, holding, injection):
    """Bound on total inventory, in-progress fractions included."""
    if holding == "reserve":
        if injection == "atomic":
            return buffer
        # while the consumer waits the buffer is empty, so at most min(K, B)
        # reserved services run next to at most max(m)-1 states already taken
        return max(buffer, min(producers, buffer) + max(1, int(max(demand))) - 1)
    return buffer + producers


def tick_policy(demand, producers, buffer, latency, step_time,
                holding="hold", injection="atomic", limit=10 ** 6):
    if holding not in HOLDING or injection not in INJECTION:
        raise ValueError("unknown policy")
    m = [int(a) for a in demand]
    K, B, L, tau = int(producers), int(buffer), int(latency), int(step_time)
    if max(m) > B:
        raise ValueError("atomic demand exceeds buffer")
    stock = B
    remaining = [None] * K          # None: idle, int: ticks left
    held = []                       # producers holding a finished state (FIFO)

    def can_start():
        if holding != "reserve":
            return True
        busy = sum(1 for x in remaining if x is not None)
        return stock + busy < B

    for j in range(K):
        if can_start():
            remaining[j] = L
    t = ready = s = grabbed = 0
    while t < limit:
        if t > 0:
            for j in range(K):
                if remaining[j] is not None and j not in held:
                    remaining[j] -= 1
            for j in range(K):
                if remaining[j] == 0 and j not in held:
                    if stock < B:
                        stock += 1
                        remaining[j] = None
                    elif holding == "hold":
                        held.append(j)
                    else:               # discard
                        remaining[j] = None
        changed = True
        while changed:
            changed = False
            if s < len(m) and t >= ready:
                need = m[s] - grabbed
                if injection == "incremental" and 0 < stock < need:
                    grabbed += stock
                    stock = 0
                    changed = True
                elif stock >= need:
                    stock -= need
                    grabbed = 0
                    s += 1
                    ready = t + tau
                    changed = True
            while held and stock < B:
                j = held.pop(0)
                stock += 1
                remaining[j] = None
                changed = True
            for j in range(K):
                if remaining[j] is None and j not in held and can_start():
                    remaining[j] = L
        if s == len(m) and t >= ready:
            return t
        t += 1
    raise RuntimeError("tick limit exceeded")

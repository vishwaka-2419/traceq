"""Lane-based consumer: rotations advance independently instead of in lockstep.

The main model of the paper lets every rotation of a simultaneous group take its
next state at the same instant (atomic demand m_s). Here each running rotation is
a lane that takes one state at a time and executes for ``step_time`` after each.
Two schedulers are provided:

    'barrier'  lanes are asynchronous inside a group of the static schedule and the
               next group starts when every member of the current one has finished;
    'dynamic'  no groups: a rotation starts as soon as its non-commuting predecessors
               have finished and the width, footprint and disjoint-support rules of
               the static scheduler allow it (greedy, in index order).

Supply is the queue of the paper: K producers, FIFO buffer of capacity B, blocking
after service, completions processed before demand at equal times.
"""
from __future__ import annotations
import heapq
from collections import deque
import numpy as np

__all__ = ["rotation_graph", "simulate_lanes"]


def rotation_graph(terms):
    """Precedence lists, support masks and footprint costs of a rotation list."""
    ps = [x["pauli"] for x in terms]
    preds = [[i for i in range(j) if not ps[i].commutes(ps[j])] for j in range(len(ps))]
    return dict(predecessors=preds, supports=[p.support for p in ps],
                costs=[p.weight + 2 for p in ps])


def simulate_lanes(graph, groups, producers, buffer, step_time, services, mode="barrier",
                   states_per_rotation=69, width=4, routing_budget=24, prefill=True):
    """Run one realisation; ``services`` is a (K, n) array of producer latencies."""
    if mode not in ("barrier", "dynamic"):
        raise ValueError("mode must be 'barrier' or 'dynamic'")
    K, B, tau, r = int(producers), int(buffer), float(step_time), int(states_per_rotation)
    if K < 1 or B < 1 or tau <= 0 or r < 1:
        raise ValueError("producers, buffer, step_time and states_per_rotation must be positive")
    sv = np.asarray(services, dtype=float)
    if sv.ndim != 2 or sv.shape[0] != K or not np.all(sv > 0):
        raise ValueError("services must be a positive (producers, n) array")
    preds, supports, costs = graph["predecessors"], graph["supports"], graph["costs"]
    n_rot = len(costs)
    remaining = [r] * n_rot
    drawn = [0] * K

    def latency(p):
        i = drawn[p]
        if i >= sv.shape[1]:
            raise ValueError("service stream exhausted")
        drawn[p] = i + 1
        return sv[p, i]

    stock = B if prefill else 0
    prod = [(latency(p), p) for p in range(K)]
    heapq.heapify(prod)
    held, waiting, lanes = deque(), deque(), []
    t, consumed, finished, peak_held = 0.0, 0, 0, 0

    # ---- scheduler state
    if mode == "barrier":
        flat = sorted(j for g in groups for j in g)
        if flat != list(range(n_rot)):
            raise ValueError("groups must partition the rotations")
        gi, left = 0, len(groups[0])
        waiting.extend(groups[0])
    else:
        pending = [len(p) for p in preds]
        succ = [[] for _ in range(n_rot)]
        for j, pl in enumerate(preds):
            for i in pl:
                succ[i].append(j)
        started = [False] * n_rot
        mask = used = running = 0

        def admit():
            nonlocal mask, used, running
            for j in range(n_rot):
                if running >= width:
                    break
                if (not started[j] and pending[j] == 0 and not (mask & supports[j])
                        and used + costs[j] <= routing_budget):
                    started[j] = True
                    mask |= supports[j]; used += costs[j]; running += 1
                    waiting.append(j)
        admit()

    while finished < n_rot:
        # serve waiting lanes, then move held outputs into freed positions
        moved = True
        while moved:
            moved = False
            while stock > 0 and waiting:
                j = waiting.popleft()
                stock -= 1; consumed += 1; remaining[j] -= 1
                heapq.heappush(lanes, (t + tau, j)); moved = True
            while held and stock < B:
                p = held.popleft(); stock += 1
                heapq.heappush(prod, (t + latency(p), p)); moved = True
        tp = prod[0][0] if prod else np.inf
        tl = lanes[0][0] if lanes else np.inf
        if tp == np.inf and tl == np.inf:
            raise RuntimeError("deadlock: no pending event")
        if tp <= tl:                                   # completions before demand
            t, p = heapq.heappop(prod)
            if stock < B:
                stock += 1
                heapq.heappush(prod, (t + latency(p), p))
            else:
                held.append(p); peak_held = max(peak_held, len(held))
        else:
            # every gate that ends at this instant is handled before the scheduler
            # admits new rotations, as the static scheduler does at a group boundary
            t = lanes[0][0]
            ended = False
            while lanes and lanes[0][0] == t:
                _, j = heapq.heappop(lanes)
                if remaining[j] > 0:
                    waiting.append(j)
                    continue
                finished += 1; ended = True
                if mode == "barrier":
                    left -= 1
                else:
                    mask ^= supports[j]; used -= costs[j]; running -= 1
                    for q in succ[j]:
                        pending[q] -= 1
            if ended and mode == "barrier" and left == 0 and gi + 1 < len(groups):
                gi += 1; left = len(groups[gi]); waiting.extend(groups[gi])
            elif ended and mode == "dynamic":
                admit()
    if consumed != n_rot * r:
        raise AssertionError("state conservation failure")
    return dict(makespan=float(t), consumed=consumed, peak_held=peak_held)

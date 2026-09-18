"""Tests for the closed forms of Sections 3-4 (theory.py) and the policy bound."""
from itertools import permutations
import numpy as np
import pytest
from traceq.model import Config, simulate, sample_latencies
from traceq.analysis import aggregate_signature
from traceq.theory import (deterministic_runtime, deterministic_interval_formula,
                           fluid_runtime, fluid_interval_formula, fluid_overload_intervals,
                           histogram_envelope, cluster_family, cluster_family_runtimes,
                           minimax_floor)
from traceq.policies import tick_policy, inventory_cap, HOLDING, INJECTION


def _case(rng, smax=50, kmax=7, bmax=7):
    K = int(rng.integers(1, kmax + 1)); B = int(rng.integers(1, bmax + 1))
    m = rng.integers(0, B + 1, int(rng.integers(1, smax)))
    mu = float(rng.choice([1., 2., 3.5, 7.25, 11.])); tau = float(rng.choice([1., 2., 3.]))
    return K, B, m, mu, tau


@pytest.mark.parametrize('seed', range(40))
def test_maxplus_recursion_equals_event_simulator(seed):
    rng = np.random.default_rng(1000 + seed)
    for _ in range(25):
        K, B, m, mu, tau = _case(rng)
        pre = bool(rng.integers(0, 2))
        cfg = Config(K, B, tau, mu, 1.0, pre, 'deterministic')
        assert deterministic_runtime(m, K, B, mu, tau, prefill=pre) == pytest.approx(
            simulate(m, cfg)['makespan'], abs=1e-9)
        sv = sample_latencies(cfg, int(m.sum()) + B + 4, 0)
        sv[:, 0] = mu * (np.arange(K) + 1) / K
        assert deterministic_runtime(m, K, B, mu, tau, prefill=pre, staggered=True) == pytest.approx(
            simulate(m, cfg, services=sv)['makespan'], abs=1e-9)


@pytest.mark.parametrize('seed', range(20))
def test_interval_family_closed_form_and_sandwich(seed):
    rng = np.random.default_rng(2000 + seed)
    for _ in range(40):
        K, B, m, mu, tau = _case(rng, smax=36)
        exact = deterministic_runtime(m, K, B, mu, tau)
        assert deterministic_interval_formula(m, K, B, mu, tau) == pytest.approx(exact, abs=1e-9)
        rho = K / mu
        assert fluid_runtime(m, rho, B + K, tau) <= exact + 1e-9
        assert exact <= fluid_runtime(m, rho, B + 1, tau) + mu + 1e-9


@pytest.mark.parametrize('seed', range(10))
def test_fluid_recursion_equals_disjoint_interval_formula(seed):
    rng = np.random.default_rng(3000 + seed)
    for _ in range(60):
        C = float(rng.integers(1, 8)) + float(rng.random() < .3) * float(rng.random())
        m = rng.integers(0, int(C) + 1, int(rng.integers(1, 16)))
        rho = float(rng.uniform(.2, 3.)); tau = float(rng.choice([.5, 1., 2.]))
        t = fluid_runtime(m, rho, C, tau)
        assert fluid_interval_formula(m, rho, C, tau) == pytest.approx(t, abs=1e-9)
        fam = fluid_overload_intervals(m, rho, C, tau)
        assert sum(f['gain'] for f in fam) == pytest.approx(t - len(m) * tau, abs=1e-9)
        assert all(a['stop'] < b['start'] for a, b in zip(fam, fam[1:]))


@pytest.mark.parametrize('seed', range(12))
def test_rearrangement_envelope_exhaustive(seed):
    rng = np.random.default_rng(4000 + seed)
    for _ in range(8):
        B = int(rng.integers(2, 6)); K = int(rng.integers(1, 5))
        m = rng.integers(0, B + 1, int(rng.integers(3, 8)))
        mu = float(rng.choice([1., 2., 3.5])); tau = float(rng.choice([1., 2.])); rho = K / mu
        perms = set(permutations(m.tolist()))
        for C in (B + 1, B + K):
            if C < rho * tau:
                continue
            env = histogram_envelope(m, rho, C, tau)
            vals = [fluid_runtime(p, rho, C, tau) for p in perms]
            assert max(vals) == pytest.approx(env['upper'], abs=1e-9)
            assert fluid_runtime(sorted(m), rho, C, tau) == pytest.approx(env['upper'], abs=1e-9)
            assert fluid_runtime(sorted(m, reverse=True), rho, C, tau) == pytest.approx(env['upper'], abs=1e-9)
            assert min(vals) >= env['lower'] - 1e-9
            assert env['upper'] <= 2 * env['lower'] + 1e-9
        if B + 1 >= rho * tau:
            worst = max(deterministic_runtime(p, K, B, mu, tau) for p in perms)
            assert worst <= histogram_envelope(m, rho, B + 1, tau)['upper'] + mu + 1e-9
            assert deterministic_runtime(sorted(m, reverse=True), K, B, mu, tau) >= \
                histogram_envelope(m, rho, B + K, tau)['upper'] - 1e-9


@pytest.mark.parametrize('w', [2, 3, 4, 5, 8])
@pytest.mark.parametrize('c', [2, 3, 5, 9])
@pytest.mark.parametrize('r', [1, 2, 7])
def test_generalised_separation_family(w, c, r):
    spread, clustered = cluster_family(w, c, r)
    assert aggregate_signature(spread) == aggregate_signature(clustered)
    cfg = Config(1, w, 1, 1, 1, True, 'deterministic')
    ta, tb = cluster_family_runtimes(w, c, r)
    assert simulate(spread, cfg)['makespan'] == ta
    assert simulate(clustered, cfg)['makespan'] == tb
    assert deterministic_runtime(clustered, 1, w, 1., 1.) == tb
    # policy-independent fluid bound is tight to one time unit
    assert fluid_runtime(clustered, 1., w + 1, 1.) == pytest.approx(tb - 1, abs=1e-9)
    assert minimax_floor(ta, tb) < 1 / 3


@pytest.mark.parametrize('holding', HOLDING)
@pytest.mark.parametrize('injection', INJECTION)
def test_policy_independent_lower_bound(holding, injection):
    rng = np.random.default_rng(99)
    for _ in range(120):
        K = int(rng.integers(1, 4)); B = int(rng.integers(1, 5))
        L = int(rng.integers(1, 5)); tau = int(rng.integers(1, 4))
        m = rng.integers(0, B + 1, int(rng.integers(1, 14)))
        t = tick_policy(m, K, B, L, tau, holding, injection)
        cap = inventory_cap(m, K, B, holding, injection)
        assert t >= fluid_runtime(m, K / L, cap, tau) - 1e-9
        if holding == 'hold':
            # incremental injection never changes the makespan of the declared queue
            assert t == simulate(m, Config(K, B, tau, L, 1.0, True, 'deterministic'))['makespan']
    for w, c in ((3, 2), (4, 2), (4, 5)):
        spread, clustered = cluster_family(w, c, 3)
        assert tick_policy(spread, 1, w, 1, 1, holding, injection) == c * w * 3
        assert tick_policy(clustered, 1, w, 1, 1, holding, injection) >= ((2 * c - 1) * w - c) * 3


def test_histogram_suffices_when_peak_below_supply():
    rng = np.random.default_rng(5)
    for _ in range(200):
        m = rng.integers(0, 3, 30)               # peak 2
        rho, tau = 2.5, 1.0                      # rho*tau >= peak
        assert fluid_runtime(m, rho, 3.0, tau) == pytest.approx(30 * tau)

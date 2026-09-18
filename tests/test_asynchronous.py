"""Lane-based (asynchronous) consumer and the vectorised staged sampler."""
import numpy as np
import pytest
from traceq.model import Config, simulate, sample_latencies
from traceq.asynchronous import rotation_graph, simulate_lanes
from traceq.workloads import hubbard_rotations, schedule
from traceq.protocols import Stage, staged_moments, sample_staged_fast


def _chain(n):
    return dict(predecessors=[[j - 1] if j else [] for j in range(n)], supports=[1] * n, costs=[3] * n)


@pytest.mark.parametrize('mode', ['barrier', 'dynamic'])
@pytest.mark.parametrize('seed', range(6))
def test_single_lane_equals_the_lockstep_queue(mode, seed):
    rng = np.random.default_rng(seed)
    n, r = int(rng.integers(2, 6)), int(rng.integers(1, 9))
    K, B = int(rng.integers(1, 5)), int(rng.integers(1, 5))
    cfg = Config(K, B, float(rng.choice([1., 2., 17.])), 9.0, float(rng.choice([.08, .3, 1.])), bool(rng.integers(0, 2)), 'geometric')
    sv = sample_latencies(cfg, n * r + B + 8, 500 + seed)
    ref = simulate(np.ones(n * r, dtype=np.int64), cfg, services=sv)['makespan']
    out = simulate_lanes(_chain(n), [[j] for j in range(n)], K, B, cfg.step_time, sv, mode,
                         states_per_rotation=r, prefill=cfg.prefill)
    assert out['makespan'] == pytest.approx(ref, abs=1e-9)
    assert out['consumed'] == n * r


@pytest.mark.parametrize('mapping', ['JW', 'TT'])
def test_unconstrained_supply_reproduces_the_static_schedule(mapping):
    terms = hubbard_rotations(6, 4, mapping)
    demand, groups = schedule(terms, t_per_rotation=5)
    graph = rotation_graph(terms)
    sv = np.full((40, len(terms) * 5 + 50), 1e-6)
    for mode in ('barrier', 'dynamic'):
        out = simulate_lanes(graph, groups, 40, 40, 17.0, sv, mode, states_per_rotation=5)
        assert out['makespan'] == pytest.approx(len(demand) * 17.0, rel=1e-9)


def test_lanes_reject_bad_input():
    g = _chain(2)
    with pytest.raises(ValueError):
        simulate_lanes(g, [[0], [1]], 1, 1, 1.0, np.ones((1, 3)), 'barrier', states_per_rotation=5)
    with pytest.raises(ValueError):
        simulate_lanes(g, [[0]], 1, 1, 1.0, np.ones((1, 30)), 'barrier')
    with pytest.raises(ValueError):
        simulate_lanes(g, [[0], [1]], 1, 1, 1.0, np.ones((1, 30)), 'lockstep')


def test_vectorised_staged_sampler_matches_exact_moments():
    stages = (Stage(1, .5), Stage(3, .2), Stage(5, .8))
    mom = staged_moments(stages, reset_time=0.5)
    x = sample_staged_fast(stages, (4, 100000), 7, reset_time=0.5)
    assert x.shape == (4, 100000) and x.min() >= 9.0
    assert x.mean() == pytest.approx(mom['mean'], rel=0.01)
    assert x.var() == pytest.approx(mom['variance'], rel=0.03)
    assert np.all(sample_staged_fast((Stage(9, 1.0),), (2, 5), 1) == 9.0)


def test_compact_external_format_roundtrip(tmp_path):
    from traceq.external import load_trace, save_compact
    m = np.array([0, 7, 7, 0, 1, 255, 3], dtype=np.int64)
    save_compact(m, tmp_path / 'x.u8.xz')
    assert np.array_equal(load_trace(tmp_path / 'x.u8.xz'), m)
    (tmp_path / 'y.csv').write_text('timestep,demand\n0,2\n1,0\n2,5\n')
    assert load_trace(tmp_path / 'y.csv').tolist() == [2, 0, 5]
    with pytest.raises(ValueError):
        save_compact(np.array([256]), tmp_path / 'z.u8.xz')

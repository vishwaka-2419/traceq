import numpy as np
from traceq.protocols import Stage, staged_moments, sample_staged

def test_one_stage_recovers_geometric():
    x=staged_moments([Stage(9,.08)])
    np.testing.assert_allclose(x['mean'],112.5)
    np.testing.assert_allclose(x['variance'],9**2*(1-.08)/.08**2)

def test_early_rejection_example():
    x=staged_moments([Stage(2,.1),Stage(7,.8)])
    np.testing.assert_allclose(x['acceptance'],.08)
    np.testing.assert_allclose(x['successful_attempt_time'],9)
    np.testing.assert_allclose(x['mean'],33.75)

def test_stage_sampler_matches_moments():
    st=[Stage(2,.3),Stage(5,.7)]
    x=staged_moments(st,reset_time=1)
    a=sample_staged(st,100000,9132,reset_time=1)
    assert abs(a.mean()/x['mean']-1)<.012
    assert abs(a.var()/x['variance']-1)<.025

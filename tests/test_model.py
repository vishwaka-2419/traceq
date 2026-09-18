import numpy as np
import pytest
from traceq.model import Config, simulate, simulate_reference, sample_latencies
from traceq.analysis import fluid_buffer_requirement, certified_violation_bound, aggregate_signature, balanced_order

@pytest.mark.parametrize('w',[2,3,4,8])
@pytest.mark.parametrize('r',[1,2,16])
def test_exact_order_separation(w,r):
    A=np.tile([w]+[0]*(w-1),2*r)
    B=np.tile([w,w]+[0]*(2*w-2),r)
    c=Config(1,w,1,1,1,True,'deterministic')
    assert aggregate_signature(A)==aggregate_signature(B)
    assert simulate(A,c)['makespan']==2*w*r
    assert simulate(B,c)['makespan']==(3*w-2)*r+1

@pytest.mark.parametrize('seed',list(range(24)))
def test_independent_implementations(seed):
    rng=np.random.default_rng(seed)
    K=int(rng.integers(1,7)); B=int(rng.integers(1,8))
    m=rng.integers(0,B+1,size=int(rng.integers(1,80)))
    c=Config(K,B,float(rng.integers(1,6)),float(rng.integers(1,5)),
             float(rng.choice([.08,.3,1.])), bool(seed%2))
    services=sample_latencies(c,int(m.sum())+B+4,seed+100)
    a=simulate(m,c,services); b=simulate_reference(m,c,services)
    for k in b:
        if k=='ages': continue
        np.testing.assert_allclose(a[k],b[k],rtol=1e-12,atol=1e-9,err_msg=k)
    assert a['makespan']>=len(m)*c.step_time
    assert a['peak_inventory']<=B+K

@pytest.mark.parametrize('m',[[],[-1],[1.5],[np.nan],[[1,2],[3,4]]])
def test_bad_traces(m):
    with pytest.raises(ValueError): simulate(m,Config(1))

@pytest.mark.parametrize('kwargs',[{'producers':0},{'buffer':0},{'step_time':0},{'attempt_time':-1},{'acceptance':0},{'acceptance':1.1},{'law':'x'}])
def test_bad_config(kwargs):
    kw={'producers':1};kw.update(kwargs)
    with pytest.raises(ValueError): Config(**kw)


def test_atomic_capacity_rejection():
    with pytest.raises(ValueError): simulate([5],Config(4,4))


def test_no_demand():
    x=simulate([0,0,0],Config(2,4,step_time=1))
    assert x['makespan']==3
    assert x['consumed']==0


def test_completions_before_demands():
    x=simulate([1,1,1],Config(1,1,1,1,1,True,'deterministic'))
    np.testing.assert_array_equal(x['starts'],[0,1,2])


def test_cold_start_charged():
    x=simulate([1],Config(1,1,1,2,1,False,'deterministic'))
    assert x['makespan']==3


def test_geometric_stream_moments():
    c=Config(1)
    a=sample_latencies(c,500000,913)[0]
    assert abs(a.mean()/c.mean_latency-1)<.01
    expected=c.attempt_time**2*(1-c.acceptance)/c.acceptance**2
    assert abs(a.var()/expected-1)<.02


def test_fluid_formula_against_quadratic():
    rng=np.random.default_rng(84)
    for _ in range(200):
        m=rng.integers(0,5,30)
        rho=float(rng.uniform(0,5))
        exact=max(sum(m[i:j+1])-rho*(j-i) for j in range(len(m)) for i in range(j+1))
        assert abs(fluid_buffer_requirement(m,rho)['capacity']-exact)<1e-10


def test_balanced_histogram():
    m=np.array([1,1,1,2,2,4,4,4,4])
    assert aggregate_signature(m)==aggregate_signature(balanced_order(m))


def test_exact_binomial_zero_failure():
    assert abs(certified_violation_bound(0,1000)-(1-.05**(1/1000)))<1e-12


def test_exhausted_service_stream():
    with pytest.raises(ValueError): simulate([2,2,2],Config(1,2,1,1,1),services=[[1]])

"""Additional verification restored with the completed delivery package."""
from itertools import product
from pathlib import Path
import importlib.util
import numpy as np
import pytest
from scipy.stats import binom
from traceq.model import Config, simulate, simulate_reference
from traceq.analysis import certified_violation_bound, fluid_buffer_requirement
from traceq.protocols import Stage
from tick_oracle import tick_oracle

# 490 possible traces * 2 producer counts * 2 prefill flags * 2 step durations
# * 3 deterministic stream patterns = 11,760 independently clocked cases.
@pytest.mark.parametrize('B',[1,2,3])
@pytest.mark.parametrize('K',[1,2])
@pytest.mark.parametrize('prefill',[False,True])
@pytest.mark.parametrize('tau',[1,2])
def test_clock_tick_oracle_exhaustive(B,K,prefill,tau):
    for length in range(1,5):
        for demand in product(range(B+1),repeat=length):
            n=sum(demand)+B+8
            for kind in range(3):
                services=np.array([[1 if kind==0 else 2 if kind==1 else 1+(j+q)%3
                                    for q in range(n)] for j in range(K)],dtype=float)
                config=Config(K,B,tau,1,1,prefill,'deterministic')
                expected=tick_oracle(demand,config,services)
                for actual in (simulate(demand,config,services),simulate_reference(demand,config,services)):
                    for key,value in expected.items():
                        np.testing.assert_allclose(actual[key],value,rtol=0,atol=1e-10,err_msg=str((B,K,prefill,tau,demand,kind,key)))


def test_validation_figure_uses_family_adjusted_bounds(monkeypatch):
    path=Path(__file__).resolve().parents[1]/'scripts/make_figures.py'
    spec=importlib.util.spec_from_file_location('traceq_figure_module',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    captured={}
    def intercept(fig,name):
        captured['heights']=[p.get_height() for p in fig.axes[0].patches]
        module.plt.close(fig)
    monkeypatch.setattr(module,'save',intercept)
    module.fig_validation()
    rows=sorted(((k,r) for k,r in module.RES.items() if k.startswith('validation_')),
                key=lambda kv:(kv[0].split('_')[1],int(kv[0].split('_')[3])))
    expected=[r['validation']['violation_ucb_family'] for _,r in rows]
    np.testing.assert_allclose(captured['heights'],expected,rtol=0,atol=1e-15)
    assert any(abs(r['stats']['violation_ucb95']-v)>1e-6 for (_,r),v in zip(rows,expected))

@pytest.mark.parametrize('n',[1,2,10,100])
def test_clopper_pearson_coverage_by_enumeration(n):
    for confidence in [.95,1-.05/8]:
        upper=np.array([certified_violation_bound(x,n,confidence) for x in range(n+1)])
        for p in np.linspace(0,1,101):
            coverage=binom.pmf(np.arange(n+1),n,p)[upper+1e-14>=p].sum()
            assert coverage>=confidence-1e-12

@pytest.mark.parametrize('rate',[0.,.5,1.,2.])
def test_fluid_capacity_is_feasible_and_tight(rate):
    for m in product(range(3),repeat=4):
        capacity=fluid_buffer_requirement(m,rate)['capacity']
        def feasible(B):
            b=B
            for i,x in enumerate(m):
                if i: b=min(B,b+rate)
                if x>b+1e-10:return False
                b-=x
            return True
        assert feasible(capacity)
        if capacity>0:assert not feasible(capacity-1e-5)

@pytest.mark.parametrize('case',range(8))
def test_additional_invalid_input_rejection(case):
    with pytest.raises(ValueError):
        if case==0:Config(1,prefill=1)
        elif case==1:Config(1,prefill='False')
        elif case==2:Stage(0,.2)
        elif case==3:Stage(1,1.1)
        elif case==4:simulate([1],Config(1),services=[[0.]])
        elif case==5:simulate([1],Config(1),services=[[float('nan')]])
        elif case==6:certified_violation_bound(.5,10)
        else:certified_violation_bound(True,10)

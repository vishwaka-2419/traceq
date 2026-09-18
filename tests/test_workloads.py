import numpy as np
import pytest
from traceq.workloads import Pauli,majoranas,hubbard_rotations,schedule,validate_schedule

@pytest.mark.parametrize('mapping',['JW','TT'])
@pytest.mark.parametrize('n',[1,4,48])
def test_majorana_algebra(mapping,n):
    g=majoranas(n,mapping)
    assert len(g)==2*n
    for i,p in enumerate(g):
        assert p.hermitian()
        assert p*p==Pauli(0,0,0)
        for q in g[:i]: assert not p.commutes(q)

@pytest.mark.parametrize('mapping,steps,peak',[('JW',8073,4),('TT',7728,3)])
def test_workload_counts(mapping,steps,peak):
    terms=hubbard_rotations(mapping=mapping)
    m,groups=schedule(terms)
    assert len(terms)==176
    assert len(m)==steps
    assert m.sum()==12144
    assert m.max()==peak
    assert validate_schedule(terms,groups,24,4)


def matrix(p,n):
    I=np.eye(2);X=np.array([[0,1],[1,0]]);Y=np.array([[0,-1j],[1j,0]]);Z=np.diag([1,-1])
    out=np.array([[1.]])
    for char in p.characters(n)[::-1]:
        out=np.kron(out,{'I':I,'X':X,'Y':Y,'Z':Z}[char])
    return p.canonical_sign()*out


def test_hubbard_against_explicit_fermion_operators():
    # 2 sites, 4 orbitals: exact 16-dimensional operator comparison.
    n=4;g=majoranas(n,'JW')
    a=[(matrix(g[2*j],n)+1j*matrix(g[2*j+1],n))/2 for j in range(n)]
    I=np.eye(2**n)
    H=np.zeros_like(I,dtype=complex)
    for u,v in [(0,2),(1,3)]:
        H-=a[u].conj().T@a[v]+a[v].conj().T@a[u]
    for u,v in [(0,1),(2,3)]:
        H+=4*(a[u].conj().T@a[u]-.5*I)@(a[v].conj().T@a[v]-.5*I)
    terms=hubbard_rotations(2,1,'JW')
    mapped=sum(t['coefficient']*matrix(t['pauli'],n) for t in terms)
    np.testing.assert_allclose(mapped,H,atol=1e-12)
    # Different faithful CAR representations must have the same spectrum.
    other=sum(t['coefficient']*matrix(t['pauli'],n) for t in hubbard_rotations(2,1,'TT'))
    np.testing.assert_allclose(np.linalg.eigvalsh(other),np.linalg.eigvalsh(H),atol=1e-12)

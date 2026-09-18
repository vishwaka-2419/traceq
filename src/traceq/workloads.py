"""Auditable Pauli-rotation demand proxies for the shifted Hubbard Hamiltonian.

This is not a placement-and-routing compiler or a certified Clifford+T
synthesis implementation. It generates legal abstract orderings and applies
a declared per-rotation sequential-T budget. All schedule choices are exported.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from collections import deque
import json
import numpy as np

@dataclass(frozen=True)
class Pauli:
    # Operator convention: i**phase * X**x * Z**z; x,z are bit masks.
    x: int
    z: int
    phase: int = 0
    def __mul__(self, other):
        return Pauli(self.x^other.x,self.z^other.z,
                     (self.phase+other.phase+2*(self.z&other.x).bit_count())%4)
    @property
    def support(self): return self.x|self.z
    @property
    def weight(self): return self.support.bit_count()
    def commutes(self, other):
        return ((self.x&other.z).bit_count()+(self.z&other.x).bit_count())%2==0
    def hermitian(self):
        return (self.phase-(self.x&self.z).bit_count())%2==0
    def characters(self,n):
        return ''.join('I' if not ((self.support>>j)&1) else
                       'Y' if ((self.x>>j)&1) and ((self.z>>j)&1) else
                       'X' if (self.x>>j)&1 else 'Z' for j in range(n))
    def canonical_sign(self):
        return (1j)**((self.phase-(self.x&self.z).bit_count())%4)


def majoranas(n: int, mapping: str):
    if n<1: raise ValueError('n must be positive')
    if mapping=='JW':
        out=[]
        for j in range(n):
            before=(1<<j)-1
            out.extend([Pauli(1<<j,before),Pauli(1<<j,before|(1<<j),1)])
        return out
    if mapping!='TT': raise ValueError('mapping must be JW or TT')
    # Breadth-first expansion of the shallowest lexicographic leaf.
    leaves=[((),Pauli(0,0))]
    for q in range(n):
        leaves.sort(key=lambda a:(len(a[0]),a[0]))
        path,p=leaves.pop(0)
        for j,(x,z,phase) in enumerate([(1,0,0),(1,1,1),(0,1,0)]):
            leaves.append((path+(j,),p*Pauli(x<<q,z<<q,phase)))
    leaves.sort(key=lambda a:a[0])
    return [p for _,p in leaves[:2*n]]  # omit last of 2n+1 anticommuting leaves


def hubbard_rotations(nx=6,ny=4,mapping='JW',hopping=1.,interaction=4.):
    if nx<1 or ny<1: raise ValueError('positive grid sizes required')
    n=2*nx*ny
    g=majoranas(n,mapping)
    terms=[]
    def add(indices, coeff, label):
        p=Pauli(0,0)
        for i in indices: p=p*g[i]
        c=complex(coeff)*p.canonical_sign()
        if abs(c.imag)>1e-12: raise AssertionError('Hamiltonian term not Hermitian')
        # Store canonical tensor-Pauli sign in the scalar coefficient.
        p=Pauli(p.x,p.z,(p.x&p.z).bit_count()%4)
        terms.append(dict(pauli=p,coefficient=float(c.real),label=label))
    for y in range(ny):
        for x in range(nx):
            site=y*nx+x
            for xx,yy in [(x+1,y),(x,y+1)]:
                if xx>=nx or yy>=ny: continue
                other=yy*nx+xx
                for spin in range(2):
                    u,v=2*site+spin,2*other+spin
                    add([2*u,2*v+1],-0.5j*hopping,f'hop:{u}:{v}:a')
                    add([2*u+1,2*v],0.5j*hopping,f'hop:{u}:{v}:b')
    for site in range(nx*ny):
        u,v=2*site,2*site+1
        add([2*u,2*u+1,2*v,2*v+1],-interaction/4,f'onsite:{site}')
    return terms


def schedule(terms,routing_budget=24,width=4,t_per_rotation=69):
    if width<1 or t_per_rotation<1: raise ValueError('positive budgets required')
    ps=[x['pauli'] for x in terms]
    if any(p.weight+2>routing_budget for p in ps):
        raise ValueError('routing budget too small for an individual rotation')
    predecessors=[{i for i in range(j) if not ps[i].commutes(ps[j])} for j in range(len(ps))]
    done=set(); groups=[]
    while len(done)<len(ps):
        ready=[j for j in range(len(ps)) if j not in done and predecessors[j]<=done]
        group=[]; mask=0; used=0
        for j in ready:
            cost=ps[j].weight+2
            if len(group)<width and not (mask&ps[j].support) and used+cost<=routing_budget:
                group.append(j); mask|=ps[j].support; used+=cost
        if not group: raise AssertionError('no schedulable ready rotation')
        groups.append(group); done.update(group)
    demand=np.repeat(np.array([len(g) for g in groups],np.int64),t_per_rotation)
    validate_schedule(terms,groups,routing_budget,width)
    return demand,groups


def validate_schedule(terms,groups,routing_budget,width):
    ps=[x['pauli'] for x in terms]
    flat=[j for group in groups for j in group]
    if sorted(flat)!=list(range(len(ps))): raise AssertionError('missing/duplicated term')
    layer={j:s for s,g in enumerate(groups) for j in g}
    for j in range(len(ps)):
        for i in range(j):
            if not ps[i].commutes(ps[j]) and not layer[i]<layer[j]:
                raise AssertionError('noncommuting precedence violated')
    for group in groups:
        if len(group)>width: raise AssertionError('injection width violated')
        if sum(ps[j].weight+2 for j in group)>routing_budget: raise AssertionError('routing budget violated')
        support=0
        for j in group:
            if support&ps[j].support: raise AssertionError('overlapping support')
            support|=ps[j].support
    return True


def export_workload(path,nx=6,ny=4,mapping='JW',routing_budget=24,width=4,t_per_rotation=69):
    from pathlib import Path
    from .analysis import aggregate_signature
    path=Path(path); path.mkdir(parents=True,exist_ok=True)
    terms=hubbard_rotations(nx,ny,mapping)
    demand,groups=schedule(terms,routing_budget,width,t_per_rotation)
    name=f'hubbard_{nx}x{ny}_{mapping}'
    np.savetxt(path/(name+'.csv'),np.c_[np.arange(len(demand)),demand],fmt='%d',delimiter=',',header='step,states',comments='')
    meta=dict(name=name,nx=nx,ny=ny,mapping=mapping,routing_budget=routing_budget,width=width,
              t_per_rotation=t_per_rotation,groups=groups,signature=aggregate_signature(demand),
              terms=[dict(label=t['label'],coefficient=t['coefficient'],x=t['pauli'].x,z=t['pauli'].z,
                          phase=t['pauli'].phase,pauli=t['pauli'].characters(2*nx*ny),weight=t['pauli'].weight) for t in terms])
    (path/(name+'.json')).write_text(json.dumps(meta,indent=2)+'\n')
    return demand,meta

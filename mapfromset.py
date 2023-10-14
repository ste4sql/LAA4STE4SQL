from bipartite_matching import bipartite_matching
import math, sys, copy
from random import sample
from utils import log, cross_likelihood, dictify
from bipartite_matching import bipartite_matching
from poaa import poaa


def mapfromset(S,aux1,aux2,obs_freqs,u1,u2,eps):
    assert(len(S) == len(obs_freqs))
    if u1 == 0 and u2 == 0:
        f = bipartite_matching(aux1,aux2,obs_freqs)
        p = cross_likelihood(aux1,aux2,obs_freqs,u1,u2,f,set(),set())
        return f, set(), set(), p
    t1 = 0
    t2 = 0
    aux1_dict = dictify(aux1)
    aux2_dict = dictify(aux2)
    for v in S:
        t1 += aux1_dict[v]
        t2 += aux2_dict[v]
    aux1_prime = []
    aux2_prime = []
    for v in S:
        aux1_prime.append((aux1_dict[v] / t1, v))
        aux2_prime.append((aux2_dict[v] / t2, v))
    f = bipartite_matching(aux1_prime,aux2_prime,obs_freqs)

    vals = set(aux1_dict.keys())
    R = vals.difference(set(S))
    U1, U2 = poaa(R,aux1,aux2,u1,u2,eps)
    p = cross_likelihood(aux1,aux2,obs_freqs,u1,u2,f,U1,U2)
    return f, U1, U2, p
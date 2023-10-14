import sys
import math
from selattack import selattack
from poaa import poaa
from utils import log, cross_likelihood


def no_cross(aux1,aux2,obs_freqs,u1,u2,eps):
    vals = set([v for (f,v) in aux1])
    obs_freqs1 = []
    obs_freqs2 = []
    for (f1,f2) in obs_freqs:
        if f1 > 0:
            obs_freqs1.append(f1)
        if f2 > 0:
            obs_freqs2.append(f2)
    obs_freqs1 = sorted(obs_freqs1) + [u1]
    obs_freqs2 = sorted(obs_freqs2) + [u2]
    f1 = selattack(aux1,obs_freqs1,eps)
    f2 = selattack(aux2,obs_freqs2,eps)
    if u1 > 0 or u2 > 0:
        R1 = vals.difference(set(f1))
        R2 = vals.difference(set(f2))
        U1_1,U2_1 = poaa(R1,aux1,aux2,u1,u2,eps)
        U1_2,U2_2 = poaa(R2,aux1,aux2,u1,u2,eps)
    else:
        U1_1 = U2_1 = U1_2 = U2_2 = set()

    if cross_likelihood(aux1,aux2,obs_freqs,u1,u2,f1,U1_1,U2_1) >= cross_likelihood(aux1,aux2,obs_freqs,u1,u2,f2,U1_2,U2_2):
        (f,U1,U2) = (f1,U1_1,U2_1)
    else:
        (f,U1,U2) = (f2,U1_2,U2_2)
    return f, U1, U2
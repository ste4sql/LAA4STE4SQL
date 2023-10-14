import copy
from math import ceil
from utils import log, dictify

def poaa(R,dist1,dist2,u1,u2,eps):
    if u1 == 0:
        return set(), R
    if u2 == 0:
        return R, set()
    if not isinstance(dist1, dict):
        dist1 = dictify(dist1)
    if not isinstance(dist2, dict):
        dist2 = dictify(dist2)
    t1 = 0
    t2 = 0
    R = list(R)
    for v in R:
        t1 += dist1[v]
        t2 += dist2[v]
    rnd = lambda x : round((x / t1) / eps)
    T = dict()
    T[rnd(dist1[R[0]])] = (0,{R[0]},1)
    T[0] = (dist2[R[0]],set(),1)
    for i in range(1,len(R)):
        K = list(T.keys())
        for x in K:
            (y,S,c) = T[x]
            if c != i:
                continue
            left = rnd(t1*eps*x+dist1[R[i]])
            if left in T.keys():
                (y2,S2,c2) = T[left]
                if y > y2:
                    T[left] = (y,S.union({R[i]}),c+1)
            else:
                T[left] = (y,S.union({R[i]}),c+1)
            T[x] = (y+dist2[R[i]],S,c+1)

    bestS = set()
    m = 0
    for x in T.keys():
        (y,S,c) = T[x]
        if c != len(R):
            continue
        s1 = sum([dist1[v] for v in S])
        s2 = sum([dist2[v] for v in set(R).difference(S)])
        if u1*log(s1) + u2*log(s2) > m:
            bestS = S
            m = u1*log(s1) + u2*log(s2)
    return bestS, set(R).difference(bestS)
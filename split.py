from bipartite_matching import bipartite_matching
from poaa import poaa
from utils import cross_likelihood
import copy, sys


# takes dist1 and dist2 as dictionaries with keys 0,..,(n-m)-1
def split_dist(aux1,aux2,obs_freqs,u1,u2,eps,dist1,dist2):
    vals = [v for (f,v) in aux1]
    n = len(vals)
    m = len(obs_freqs)
    S1,S2 = poaa(range(n-m),dist1,dist2,u1,u2,eps)
    new_freqs = copy.deepcopy(obs_freqs)
    for i in range(n-m):
        if i in S1:
            f1 = (u1 + u2) * dist1[i]
            f2 = sys.float_info.min # 0 causes bipartite matching to sometimes err
        elif i in S2:
            f1 = sys.float_info.min # 0 causes bipartite matching to sometimes err
            f2 = (u1 + u2) * dist2[i]
        else:
            f1 = 0
            f2 = 0
        new_freqs.append((f1,f2))
    g = bipartite_matching(aux1,aux2,new_freqs)
    f = g[:m] # matching for the first m values
    U1 = set()
    U2 = set()
    for i in range(n-m):
        if i in S1:
            U1.add(g[m+i])
        if i in S2:
            U2.add(g[m+i])
    return f, U1, U2


def split(aux1,aux2,obs_freqs,u1,u2,eps):
    vals = [v for (f,v) in aux1]
    n = len(vals)
    m = len(obs_freqs)
    if n == m:
        f = bipartite_matching(aux1,aux2,obs_freqs)
        return f, set(), set()

    unif = dict()
    zipf = dict()
    aux1_dict = dict()
    aux2_dict = dict()
    Hn = sum([1/i for i in range(1,(n-m)+1)])
    sorted_aux1 = sorted(aux1,reverse=True)
    sorted_aux2 = sorted(aux2,reverse=True)
    for i in range(n-m):
        unif[i] = 1 / (n-m)
        zipf[i] = 1 / ((i+1) * Hn)
        aux1_dict[i] = sorted_aux1[m+i][0]
        aux2_dict[i] = sorted_aux2[m+i][0]

    first_dists = [unif, zipf, aux1_dict]
    second_dists = [unif, zipf, aux2_dict]

    best_p = None
    for dist1 in first_dists:
        for dist2 in second_dists:
            f,U1,U2 = split_dist(aux1,aux2,obs_freqs,u1,u2,eps,dist1,dist2)
            p = cross_likelihood(aux1,aux2,obs_freqs,u1,u2,f,U1,U2)
            if best_p == None or p > best_p:
                best_p,best_f,best_U1,best_U2 = p,f,U1,U2
    return best_f, best_U1, best_U2

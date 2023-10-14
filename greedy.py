from mapfromset import mapfromset
import math, sys, copy
from random import seed, uniform, sample


## Naive implementation of greedy algorithm
def greedy(aux1,aux2,obs_freqs,u1,u2,eps):
    vals = [v for (f,v) in aux1]
    if u1 == 0 and u2 == 0:
        f,U1,U2,_ = mapfromset(set(vals),aux1,aux2,obs_freqs,u1,u2,eps)
        return f,U1,U2
    M = len(obs_freqs)
    cap = math.ceil(1/eps)

    def optimize(starting_set):
        curr_set = starting_set
        delta = 1
        iters = 0
        f_best,U1_best,U2_best,p_best = mapfromset(curr_set,aux1,aux2,obs_freqs,u1,u2,eps)
        while delta > 0 and iters <= cap:
            delta = 0
            iters += 1
            prev_set = copy.deepcopy(curr_set)
            outside_set = set(vals).difference(prev_set)
            for v in prev_set:
                for u in outside_set:
                    # remove v from set and add u
                    S = prev_set.difference(set([v])).union(set([u]))
                    f,U1,U2,p = mapfromset(S,aux1,aux2,obs_freqs,u1,u2,eps)
                    if p - p_best > delta:
                        curr_set = S
                        f_best,U1_best,U2_best,p_best = f,U1,U2,p
                        delta = p - p_best
        return f_best,U1_best,U2_best,p_best

    best_soln = optimize(set(sample(vals,M)))
    for _ in range(5): # just hard coding 5 for now
        random_set = set(sample(vals,M))
        soln = optimize(random_set)
        # compare likelihoods
        if soln[-1] > best_soln[-1]:
            best_soln = soln
    (f,U1,U2,_) = best_soln
    return f, U1, U2
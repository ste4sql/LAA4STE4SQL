import math, sys, copy
from utils import log

sys.setrecursionlimit(10000000)

# aux_freqs is a list of (freq, val)
# assume obs_freqs is sorted in ascending order
def selattack(aux_dist, obs_freqs, eps):
    N, M = len(aux_dist), len(obs_freqs) - 1
    aux_dist = sorted(aux_dist,key=lambda x: x[0])
    if N == M:
        ## return frequency analysis result
        return [aux_dist[i][1] for i in range(N)]
    aux_freqs = [f for (f,v) in aux_dist]
    tail_sums = [sum(aux_freqs[:i]) for i in range(N)]

    memo = {}
    nxt = {}
    rnd = lambda x : round(x / eps)

    # returns an assignment
    def helper(i, j, sigma):
        try: return memo[i, j, sigma]
        except:
            if j == 0:
                prob = obs_freqs[M] * log(sigma*eps + tail_sums[i])
            elif i < j:
                print(nxt)
                print(memo)
                assert(False)
            elif i == j:
                nxt[i, j, sigma] = (i-1, j-1, sigma)
                prob = helper(i-1, j-1, sigma) + obs_freqs[j-1] * log(aux_freqs[i-1])
            else:
                p1 = helper(i-1, j-1, sigma) + obs_freqs[j-1] * log(aux_freqs[i-1])
                sp = rnd(sigma*eps + aux_freqs[i-1])
                p2 = helper(i-1, j, sp)
                prob = max(p1,p2)
                if p1 >= p2:
                    nxt[i, j, sigma] = (i-1, j-1, sigma)
                else:
                    nxt[i, j, sigma] = (i-1, j, sp)
            memo[i, j, sigma] = prob
            return copy.deepcopy(prob)
    
    prob = helper(N, M, 0)
    assignment = []
    i, j, sigma = N, M, 0
    while True:
        if j == 0: break
        elif i < j: assert(False)
        else:
            if nxt[i, j, sigma] == (i-1, j-1, sigma):
                assignment = [aux_dist[i-1][1]] + assignment
            i, j, sigma = nxt[i, j, sigma]
    assert(len(assignment) == M)
    return assignment
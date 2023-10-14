import math
import numpy as np
from scipy.optimize import linear_sum_assignment
from utils import log, dictify
import sys

def bipartite_matching(aux1, aux2, obs_freqs):
    assert(len(aux1) == len(obs_freqs))
    obs1 = []
    obs2 = []
    for (f1,f2) in obs_freqs:
        obs1.append(f1)
        obs2.append(f2)
    assert(len(aux1) == len(obs1))
    assert(len(aux2) == len(obs2))
    aux1 = dictify(aux1)
    aux2 = dictify(aux2)
    N = len(obs1)

    ordered_keys = list(aux1.keys())
    matrix = [
        [obs1[y] * log(aux1[v]) + obs2[y] * log(aux2[v])
        for v in ordered_keys] for y in range(N)
    ]

    # remove negative infinities
    for i in range(N):
        for j in range(len(ordered_keys)):
            if matrix[i][j] == float('-inf'):
                matrix[i][j] = -1 * sys.float_info.max
            if np.isnan(matrix[i][j]):
                if aux1[ordered_keys[j]] == 0 or aux2[ordered_keys[j]] == 0:
                    matrix[i][j] = -1 * sys.float_info.max
                else:
                    # throw error, because idk what caused a nan
                    print("unknown entry at ", i, j)
                    return

    # returns a mapping from rows[i] -> cols[i] that sum of weights
    # this corresponds to obs_freq[i] -> ordered_key[i] and rows are sorted
    rows, cols = linear_sum_assignment(cost_matrix=np.array(matrix),maximize=True)

    val_match = [ordered_keys[i] for i in cols]
    return val_match
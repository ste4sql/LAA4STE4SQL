import sys, math

def log(x):
    if x == 0: return float("-inf")
    else: return math.log(x)

# convert (frequency, val) pair to dictionary
def dictify(dist):
    new_dist = dict()
    for (f,v) in dist:
        new_dist[v] = f
    return new_dist

# convert dictionary distribution into not dictionary
def undictify(dist):
    new_dist = []
    for (v,f) in dist.keys():
        new_dist.append((f,v))
    return new_dist

def cross_likelihood(aux1,aux2,obs_freqs,u1,u2,f,U1,U2):
    assert(len(obs_freqs) == len(f))
    aux1_dict = dictify(aux1)
    aux2_dict = dictify(aux2)
    s1 = 0
    for v in U1:
        s1 += aux1_dict[v]
    s2 = 0
    for v in U2:
        s2 += aux2_dict[v]
    p = u1 * log(s1) + u2 * log(s2)
    for i in range(len(obs_freqs)):
        (c1,c2) = obs_freqs[i]
        p = p + c1*log(aux1_dict[f[i]]) + c2 * log(aux2_dict[f[i]])
    return p
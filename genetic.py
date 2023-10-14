from mapfromset import mapfromset
from math import ceil
from random import seed, randint, sample

pop_size = 30 # want to be at least 10
generations = 15
tournament_size = 5
mutation_rate = 0.2

def tournament(t,n):
    return min([randint(0,n-1) for _ in range(t)])


def mutate(mutations,S,vals):
    c = randint(1,mutations)
    U = set(sample(list(S),c))
    # sample from vals\S and U to make sure there are c values available
    diff = (set(vals).difference(set(S))).union(U)
    V = set(sample(list(diff),c))
    return set(S).difference(U).union(V)

## genetic algorithm with parameters set above
def genetic(aux1,aux2,obs_freqs,u1,u2,eps):
    vals = [v for (f,v) in aux1]
    n = len(vals)
    m = len(obs_freqs)
    mutations = ceil(mutation_rate * m)

    top_keeps = ceil(pop_size / 10)
    mutate_keeps = round(pop_size * 4 / 5)
    if top_keeps + mutate_keeps >= pop_size:
        keep = max((top_keeps - mutate_keeps) - 1, 0)
    random_keeps = pop_size - (top_keeps + mutate_keeps)


    population = [set(sample(vals,m)) for _ in range(pop_size)]
    for _ in range(generations):
        maps = []
        for S in population:
            f,U1,U2,p = mapfromset(S,aux1,aux2,obs_freqs,u1,u2,eps)
            maps.append((S,f,U1,U2,p))
        maps = sorted(maps,key=lambda x: x[-1],reverse=True) # sort by most to least likely

        # keep the best ~10% of sets
        population = [S for (S,f,U1,U2,p) in maps[:top_keeps]]

        # pick winners and mutate winners
        for _ in range(mutate_keeps):
            idx = tournament(tournament_size,pop_size)
            mutated = mutate(mutations,maps[idx][0],vals)
            population.append(mutated)

        # randomly keep some of the population
        for _ in range(random_keeps):
            samp = randint(0,pop_size-1)
            population.append(maps[samp][0])
    # select the best in the population
    for S in population:
        f,U1,U2,p = mapfromset(S,aux1,aux2,obs_freqs,u1,u2,eps)
        maps.append((S,f,U1,U2,p))
    maps = sorted(maps,key=lambda x: x[-1],reverse=True) # sort by most to least likely
    S,f,U1,U2,p = maps[0]
    return f, U1, U2

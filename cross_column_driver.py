import sys
import math
from pandas import read_csv

from greedy import greedy
from split import split
from genetic import genetic
from no_cross import no_cross

eps = 0.001
output_file = open('cross_2018_exps.csv','w')
# output_file = None

# returns the auxiliary distribution and the histogram for the observed data
def hists_from_files(exp_params):
    (_,col1,col2) = exp_params
    (aux_fname1,obs_fname1,col_name1) = col1
    (aux_fname2,obs_fname2,col_name2) = col2
    aux_col1 = read_csv(aux_fname1, sep=',', dtype=str, keep_default_na=False, usecols=[col_name1])[col_name1]
    aux_col2 = read_csv(aux_fname2, sep=',', dtype=str, keep_default_na=False, usecols=[col_name2])[col_name2]
    obs_col1 = read_csv(obs_fname1, sep=',', dtype=str, keep_default_na=False, usecols=[col_name1])[col_name1]
    obs_col2 = read_csv(obs_fname2, sep=',', dtype=str, keep_default_na=False, usecols=[col_name2])[col_name2]
    aux_freqs1 = aux_col1.value_counts(normalize=True)
    aux_dist1 = [(aux_freqs1[i],aux_freqs1.index[i])  for i in range(len(aux_freqs1))]
    aux_freqs2 = aux_col2.value_counts(normalize=True)
    aux_dist2 = [(aux_freqs2[i],aux_freqs2.index[i])  for i in range(len(aux_freqs2))]


    obs_freqs1 = obs_col1.value_counts()
    obs_hist1 = [(obs_freqs1[i],obs_freqs1.index[i])  for i in range(len(obs_freqs1))]
    obs_freqs2 = obs_col2.value_counts()
    obs_hist2 = [(obs_freqs2[i],obs_freqs2.index[i])  for i in range(len(obs_freqs2))]
    return aux_dist1, aux_dist2, obs_hist1, obs_hist2


# assignment is a list of indices to guess
def get_scores(correct_vals,guessed_vals,obs_freqs):
    # change r-score to be the fraction of rows correctly
    # identified over the ones observed
    assert(len(correct_vals) == len(guessed_vals))
    assert(len(obs_freqs) == len(guessed_vals))
    rscore = 0
    vscore = 0
    M = len(guessed_vals)
    obs_rows = 0
    for i in range(M):
        f1, f2 = obs_freqs[i]
        obs_rows = obs_rows + f1 + f2
        if correct_vals[i] == guessed_vals[i]:
            rscore = rscore + f1 + f2
            vscore += 1
    rscore = rscore / obs_rows
    vscore = vscore / M
    return rscore, vscore

# runs is number of runs for random experiments and 0 for deterministic ones
def write_scores(exp_params,alg,u1,u2,u3,n1,n2,m,N,rscore,vscore):
    (exp_name,col1,col2) = exp_params
    (year, category) = exp_name
    data = [year,category,alg,u1,u2,u3,n1,n2,m,N,100*rscore,100*vscore]
    line = ','.join([str(d) for d in data])
    if output_file is not None:
        output_file.write(line + '\n')
        output_file.flush()
    else:
        print(line)



def run_attacks(exp_params):
    aux1, aux2, obs1, obs2 = hists_from_files(exp_params)

    aux1_vals = set([v for (f,v) in aux1])
    obs1_vals = set([v for (f,v) in obs1])
    aux2_vals = set([v for (f,v) in aux2])
    obs2_vals = set([v for (f,v) in obs2])
    vals = aux1_vals.union(obs1_vals).union(aux2_vals).union(obs2_vals)
    for v in vals:
        if not v in aux1_vals:
            aux1.append((0,v))
        if not v in obs1_vals:
            obs1.append((0,v))
        if not v in aux2_vals:
            aux2.append((0,v))
        if not v in obs2_vals:
            obs2.append((0,v))

    obs1 = sorted(obs1,key=lambda x: x[0]) # make sure we have the sorted invariant
    N = len(aux1)

    # create list of frequencies
    obs_freqs = []
    correct_vals = []
    u1 = 0
    u2 = 0
    u3 = 0
    n1 = 0
    n2 = 0
    m = 0
    for (f1,v) in obs1:
        for (f2,v2) in obs2:
            if v == v2:
                if f1 > 0 and f2 > 0:
                    correct_vals.append(v)
                    obs_freqs.append((f1,f2))
                    n1 += 1
                    n2 += 1
                    m += 1
                elif f1 > 0 and f2 == 0:
                    u1 += f1
                    n1 += 1
                elif f2 > 0 and f1 == 0:
                    u2 += f2
                    n2 += 1
                else:
                    u3 += 1

    # test the baseline
    f,U1,U2 = baseline(aux1,aux2,obs_freqs,u1,u2,eps)
    rscr, vscr = get_scores(correct_vals,f,obs_freqs)
    write_scores(exp_params,"baseline",u1,u2,u3,n1,n2,m,N,rscr,vscr)


    # test greedy
    f,U1,U2 = greedy(aux1,aux2,obs_freqs,u1,u2,eps)
    rscr, vscr = get_scores(correct_vals,f,obs_freqs)
    write_scores(exp_params,"greedy",u1,u2,u3,n1,n2,m,N,rscr,vscr)


    # test genetic
    f,U1,U2 = genetic(aux1,aux2,obs_freqs,u1,u2,eps)
    rscr, vscr = get_scores(correct_vals,f,obs_freqs)
    write_scores(exp_params,"genetic",u1,u2,u3,n1,n2,m,N,rscr,vscr)


    # test split
    f,U1,U2 = split(aux1,aux2,obs_freqs,u1,u2,eps)
    rscr, vscr = get_scores(correct_vals,f,obs_freqs)
    write_scores(exp_params,"split",u1,u2,u3,n1,n2,m,N,rscr,vscr)





if __name__ == "__main__":
    crimes = dict()
    taxi = dict()
    crashes = dict()
    rideshares = dict()
    for year in range(2018,2023):
        crimes[year] = "../data/chicago/crimes-"+ str(year) + ".csv"
        taxi[year] = "../data/chicago/taxi-"+ str(year) + ".csv"
        crashes[year] = "../data/chicago/crashes-"+ str(year) + ".csv"
        rideshares[year] = "../data/chicago/rideshares-"+ str(year) + ".csv"

    taxi_pickup = dict()
    taxi_dropoff = dict()
    crimes_community = dict()
    crimes_beat = dict()
    crashes_beat = dict()
    rideshares_pickup = dict()
    rideshares_dropoff = dict()
    aux_year = 2018
    for year in range(2019,2023):
        taxi_pickup[year] = (taxi[aux_year],taxi[year],"Pickup Community Area")
        taxi_dropoff[year] = (taxi[aux_year],taxi[year],"Dropoff Community Area")
        crimes_community[year] = (crimes[aux_year],crimes[year],"Community Area")
        crimes_beat[year] = (crimes[aux_year],crimes[year],"Beat")
        crashes_beat[year] = (crashes[aux_year],crashes[year],"BEAT_OF_OCCURRENCE")
        rideshares_pickup[year] = (rideshares[aux_year],rideshares[year],"Pickup Community Area")
        rideshares_dropoff[year] = (rideshares[aux_year],rideshares[year],"Dropoff Community Area")

    chicago_exps = []
    for year in range(2019,2023):
        chicago_exps.append(((year,"pickup-taxi-crime"), taxi_pickup[year], crimes_community[year]))
        chicago_exps.append(((year,"dropoff-taxi-crime"), taxi_dropoff[year], crimes_community[year]))
        chicago_exps.append(((year,"pickup-taxi-rideshare"), taxi_pickup[year], rideshares_pickup[year]))
        chicago_exps.append(((year,"dropoff-taxi-rideshare"), taxi_dropoff[year], rideshares_dropoff[year]))
        chicago_exps.append(((year,"pickup-rideshare-crime"), rideshares_pickup[year], crimes_community[year]))
        chicago_exps.append(((year,"dropoff-rideshare-crime"), rideshares_dropoff[year], crimes_community[year]))
        chicago_exps.append(((year,"beats"), crashes_beat[year], crimes_beat[year]))

    for exp in chicago_exps:
        run_attacks(exp)


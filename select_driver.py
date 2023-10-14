import sys
import math
from random import sample, randint
from pandas import read_csv
from selattack import selattack

eps = 0.001
fraction_observed = [0.1 * k for k in range(1,11)]
runs_for_randoms = 10
output_file = open('select_2018_exps.csv','w')

# returns the auxiliary distribution and the histogram for the observed data
def hists_from_files(exp_params):
    if len(exp_params) == 5:
        (_, aux_fname, obs_fname, aux_col, obs_col) = exp_params
    elif len(exp_params) == 4:
        (_, aux_fname, obs_fname, col_name) = exp_params
        aux_col = obs_col = col_name
    aux_col = read_csv(aux_fname, dtype=str, keep_default_na=False, usecols=[aux_col])[aux_col]
    obs_col = read_csv(obs_fname, dtype=str, keep_default_na=False, usecols=[obs_col])[obs_col]
    aux_freqs = aux_col.value_counts(normalize=True)
    aux_dist = [(aux_freqs[i],aux_freqs.index[i])  for i in range(len(aux_freqs))]
    # aux_dist = [aux_freqs[i]  for i in range(len(aux_freqs))]
    obs_freqs = obs_col.value_counts()
    obs_hist = [(obs_freqs[i],obs_freqs.index[i])  for i in range(len(obs_freqs))]
    # obs_hist = [obs_freqs[i] for i in range(len(obs_freqs))]
    return aux_dist, obs_hist


# assignment is a list of indices to guess
def get_scores(soln_samp,guess_vals,obs_hist):
    # change r-score to be the fraction of rows correctly
    # identified over the ones observed
    correct_vals = [obs_hist[i][1] for i in soln_samp]
    rscore = 0
    vscore = 0
    setscore = 0
    M = len(soln_samp)
    obs_rows = 0
    for i in range(M):
        obs_rows += obs_hist[soln_samp[i]][0]
        if correct_vals[i] == guess_vals[i]:
            rscore += obs_hist[soln_samp[i]][0]
            vscore += 1
        if correct_vals[i] in guess_vals:
            setscore += 1
    rscore = rscore / obs_rows
    vscore = vscore / M
    setscore = setscore / M
    return rscore, vscore, setscore

# runs is number of runs for random experiments and 0 for deterministic ones
def write_scores(exp_params,sampling,n,N,runs,rscore,vscore,setscore):
    if len(exp_params) == 5:
        (exp_name, aux_fname, obs_fname, aux_col, obs_col) = exp_params
        data = [exp_name,sampling,n,N,runs,rscore*100,vscore*100,setscore*100]
    elif len(exp_params) == 4:
        (exp_name, aux_fname, obs_fname, col_name) = exp_params
        (year,attribute) = exp_name
        aux_col = obs_col = col_name
        data = [year,attribute,sampling,n,N,runs,rscore*100,vscore*100,setscore*100]
    else:
        data = ['ERROR WRITING',exp_params]
    line = ','.join([str(d) for d in data])
    print(line)
    if output_file is not None:
        output_file.write(line + '\n')
        output_file.flush()



def run_selattack(exp_params):
    aux_dist, obs_hist = hists_from_files(exp_params)
    ## need to make value sets actually match up
    aux_vals = set([v for (f,v) in aux_dist])
    obs_vals = set([v for (f,v) in obs_hist])
    for v in aux_vals.union(obs_vals):
        if not v in aux_vals:
            aux_dist.append((0,v))
        if not v in obs_vals:
            obs_hist.append((0,v))
    N = len(aux_dist) ## which is now equal to obs_vals
    obs_hist = sorted(obs_hist,key=lambda x: x[0])
    obs_freqs = sorted([obs_hist[i][0] for i in range(N)])
    num_selected = set([math.ceil(N * f) for f in fraction_observed])
    for n in sorted(num_selected):
        if n > N:
            n = N
            continue
        # Allow the adversary to observe the top n entries
        samp = list(range(N-n,N))
        truncated_freqs = obs_freqs[(N-n):] + [sum(obs_freqs[:(N-n)])]
        # soln_his = obs_hist[:n]
        guess_vals = selattack(aux_dist, truncated_freqs, eps)
        rscr, vscr, sscr = get_scores(samp,guess_vals,obs_hist)
        write_scores(exp_params,"top",n,N,-1,rscr,vscr,sscr)

        # Allow the adversary to observe a random n entries
        if n == N:
            # skip extra trials rather than select all
            continue
        avg_rscr = 0
        avg_vscr = 0
        avg_sscr = 0
        for _ in range(runs_for_randoms):
            samp = sorted(sample(range(N),n))
            truncated_freqs = []
            unobserved = 0
            for idx in range(N):
                if idx in samp:
                    truncated_freqs.append(obs_freqs[idx])
                    # soln_hist.append(obs_hist)
                else:
                    unobserved += obs_freqs[idx]
            truncated_freqs = sorted(truncated_freqs) + [unobserved]
            guess_vals = selattack(aux_dist, truncated_freqs, eps)
            rscr, vscr, sscr = get_scores(samp,guess_vals,obs_hist)
            avg_rscr += rscr
            avg_vscr += vscr
            avg_sscr += sscr
        avg_rscr = avg_rscr / runs_for_randoms
        avg_vscr = avg_vscr / runs_for_randoms
        avg_sscr = avg_sscr / runs_for_randoms
        write_scores(exp_params,"rand",n,N,runs_for_randoms,rscr,vscr,sscr)

        # Allow the adversary to observe a random n entries, weighted by frequency
        avg_rscr = 0
        avg_vscr = 0
        avg_sscr = 0
        for _ in range(runs_for_randoms):
            samp = set()
            used_rows = 0
            while len(samp) < n:
                r = randint(0,sum(obs_freqs)-used_rows) # sample random row from remaining
                s = -1
                pos = 0
                while s < r and pos < N:
                    if pos in samp:
                        pos += 1
                        continue
                    s += obs_freqs[pos]
                    pos += 1
                if s >= r:
                    samp.add(pos-1)
                    used_rows += obs_freqs[pos-1]
                elif pos >= N:
                    unused = list(set(range(N)).difference(samp))
                    to_add = sample(unused,n-len(samp))
                    for pos in to_add:
                        samp.add(pos)
            samp = sorted(list(samp))
            truncated_freqs = []
            unobserved = 0
            for idx in range(N):
                if idx in samp:
                    truncated_freqs.append(obs_freqs[idx])
                else:
                    unobserved += obs_freqs[idx]
            truncated_freqs = sorted(truncated_freqs) + [unobserved]
            guess_vals = selattack(aux_dist, truncated_freqs, eps)
            rscr, vscr, sscr = get_scores(samp,guess_vals,obs_hist)
            avg_rscr += rscr
            avg_vscr += vscr
            avg_sscr += sscr
        avg_rscr = avg_rscr / runs_for_randoms
        avg_vscr = avg_vscr / runs_for_randoms
        avg_sscr = avg_sscr / runs_for_randoms
        write_scores(exp_params,"weighted",n,N,runs_for_randoms,rscr,vscr,sscr)



if __name__ == "__main__":
    # chicago experiments
    crimes = dict()
    taxi = dict()
    crashes = dict()
    rideshares = dict()
    for year in range(2018,2023):
        crimes[year] = "../data/chicago/crimes-"+ str(year) + ".csv"
        taxi[year] = "../data/chicago/taxi-"+ str(year) + ".csv"
        crashes[year] = "../data/chicago/crashes-"+ str(year) + ".csv"
        rideshares[year] = "../data/chicago/rideshares-"+ str(year) + ".csv"

    chicago_exps = []
    aux_year = 2018
    for year in range(2019,2023):
        chicago_exps.append(((year,"taxi-pickup"), taxi[aux_year],taxi[year],"Pickup Community Area"))
        chicago_exps.append(((year,"taxi-dropoff"), taxi[aux_year],taxi[year],"Dropoff Community Area"))
        chicago_exps.append(((year,"crimes-ca"), crimes[aux_year],crimes[year],"Community Area"))
        chicago_exps.append(((year,"crimes-beat"), crimes[aux_year],crimes[year],"Beat"))
        chicago_exps.append(((year,"crashes-beat"), crashes[aux_year],crashes[year],"BEAT_OF_OCCURRENCE"))
        chicago_exps.append(((year,"speed-limit"), crashes[aux_year],crashes[year],"POSTED_SPEED_LIMIT"))
        chicago_exps.append(((year,"rideshare-pickup"), rideshares[aux_year],rideshares[year],"Pickup Community Area"))
        chicago_exps.append(((year,"rideshare-dropoff"), rideshares[aux_year],rideshares[year],"Dropoff Community Area"))

    for exp in chicago_exps:
        run_selattack(exp)

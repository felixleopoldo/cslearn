import json

from numpy import flatnonzero, loadtxt

# input
dag_path = snakemake.input[0]

# load dag and find parents (poss_cvars)
dag = loadtxt(dag_path, delimiter=",")
header = dag[0]
conv = lambda nparray: [str(i) for i in nparray]
poss_cvars = {
    str(idx): conv(flatnonzero(pa_mask)) for idx, pa_mask in enumerate(dag[1:].T)
}

# output
with open(snakemake.output[0], "w") as f:
    json.dump(poss_cvars, f)

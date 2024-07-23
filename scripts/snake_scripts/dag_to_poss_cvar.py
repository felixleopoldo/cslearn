import json

from numpy import flatnonzero, loadtxt

# input
dag_path = snakemake.input[0]

# load dag and find parents (poss_cvars)
dag = loadtxt(dag_path, delimiter=",")
conv = lambda nparray: [int(i) for i in nparray]
poss_cvars = {int(idx): conv(flatnonzero(pa_mask)) for idx, pa_mask in enumerate(dag.T)}

# output
with open(snakemake.output[0], "w") as f:
    json.dump(poss_cvars, f)

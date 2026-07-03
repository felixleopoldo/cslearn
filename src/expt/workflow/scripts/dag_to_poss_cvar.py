import json

from numpy import loadtxt

# input
dag_path = snakemake.input[0]

# load dag and find parents (poss_cvars)
# read dag_path as csv file
dag = loadtxt(dag_path, delimiter=",").astype(int)
labels = dag[0]
conv = lambda nparray: [str(i) for i in nparray]
poss_cvars = {
    str(label): conv(labels[pa_mask.astype(bool)])
    for label, pa_mask in zip(labels, dag[1:].T)
}

# output
with open(snakemake.output[0], "w") as f:
    json.dump(poss_cvars, f)

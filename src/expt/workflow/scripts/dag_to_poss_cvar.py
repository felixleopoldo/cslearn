import json

from numpy import loadtxt

# input
dag_path = snakemake.input[0]

# load dag and find parents (poss_cvars)
# read dag_path as csv file
dag = loadtxt(dag_path, delimiter=",").astype(int)
labels = dag[0]


def conv(nparray):
    return [str(i) for i in nparray]


# Each column of dag[1:] is a node's parent mask; poss_cvars maps each node to its parent labels.
poss_cvars = {str(label): conv(labels[pa_mask.astype(bool)]) for label, pa_mask in zip(labels, dag[1:].T)}

# output
with open(snakemake.output[0], "w") as f:
    json.dump(poss_cvars, f)

import random

from numpy import random as nprandom

from cstrees import cstree as ct

# input from snakemake somehow?
p = p
max_cvar = max_cvar
prob_cvar = prob_cvar
seed = seed

# generate cstree
nprandom.seed(seed)
random.seed(seed)
cards = [2] * p
cstree = ct.sample_cstree(
    cards, max_cvars=max_cvar, prob_cvar=prob_cvar, prop_nonsingleton=1
)
cstree.sample_stage_parameters(alpha=2)
cstree_df = cstree.to_df(write_probs=True)

# save csv and output path with snakemake somehow?
path = f"cstrees/p={p}/max_cvar={max_cvar}/prob_cvar={prob_cvar}/seed={seed}/cstree.csv"
cstree_df.to_csv(path)

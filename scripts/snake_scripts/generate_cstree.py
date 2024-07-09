import random

from numpy import random as nprandom

from cstrees import cstree as ct

# input
p = int(snakemake.wildcards["p"])
max_cvar = int(snakemake.wildcards["max_cvar"])
prob_cvar = float(snakemake.wildcards["prob_cvar"])
seed = int(snakemake.wildcards["seed"])

# generate cstree
nprandom.seed(seed)
random.seed(seed)
cards = [2] * p
cstree = ct.sample_cstree(
    cards, max_cvars=max_cvar, prob_cvar=prob_cvar, prop_nonsingleton=1
)
cstree.sample_stage_parameters(alpha=2)
cstree_df = cstree.to_df(write_probs=True)

# output
cstree_df.to_csv(snakemake.output[0], index=False)

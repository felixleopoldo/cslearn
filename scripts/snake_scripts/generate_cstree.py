import random

from numpy import random as nprandom

from cstrees import cstree as ct

# input
p = int(snakemake.wildcards["cstree_p"])
max_cvar = int(snakemake.wildcards["cstree_max_cvar"])
prob_cvar = float(snakemake.wildcards["cstree_prob_cvar"])
seed = int(snakemake.wildcards["seed"])

# generate cstree
nprandom.seed(seed)
random.seed(seed)
cards = [2] * p
cstree = ct.sample_cstree(cards, max_cvar, prob_cvar)
cstree_df = cstree.to_df()

# output
cstree_df.to_csv(snakemake.output[0])

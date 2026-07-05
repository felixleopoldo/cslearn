import random

from numpy import random as nprandom
from pandas import read_csv

from cslearn import cstree as ct

# input
alpha = float(snakemake.wildcards["cstree_params_alpha"])
seed = int(snakemake.wildcards["seed"])
tree_df = read_csv(snakemake.input[0], index_col=(0))

# generate cstree params
nprandom.seed(seed)
random.seed(seed)
cstree = ct.df_to_cstree(tree_df)
cstree.sample_stage_parameters(alpha)
cstree_df = cstree.to_df(write_probs=True)

# output
cstree_df.to_csv(snakemake.output[0])

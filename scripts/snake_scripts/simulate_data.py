from numpy import random as nprandom
from pandas import read_csv

from cstrees import cstree as ct

# input
input_path = snakemake.input[0]
n = int(snakemake.wildcards["cstree_data_n"])
seed = int(snakemake.wildcards["seed"])

# simulate data
nprandom.seed(seed)
df = read_csv(input_path, index_col=(0))
t = ct.df_to_cstree(df)
data = t.sample(n)

# output
data.to_csv(snakemake.output[0], index=False)

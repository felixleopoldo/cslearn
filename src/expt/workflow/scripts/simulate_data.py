from numpy import random as nprandom
from pandas import read_csv

from cslearn import cstree as ct

# input
input_path = snakemake.input[0]
n_samples = int(snakemake.wildcards["cstree_data_n"])
seed = int(snakemake.wildcards["seed"])

# simulate data
nprandom.seed(seed)

df = read_csv(input_path, index_col=(0))
tree = ct.df_to_cstree(df)
data = tree.sample(n_samples)

# Resample until every variable has at least 2 observed values; required by downstream scorers.
while not all([data[col][1:].nunique() >= 2 for col in data.columns]):
    data = tree.sample(n_samples)


# output
data.to_csv(snakemake.output[0], index=False)

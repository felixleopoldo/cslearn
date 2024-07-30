from numpy import random as nprandom
from pandas import read_csv

from cstrees import cstree as ct

# input
input_path = snakemake.input[0]
n_samples = int(snakemake.wildcards["cstree_data_n"])
seed = int(snakemake.wildcards["seed"])

# simulate data
nprandom.seed(seed)

df = read_csv(input_path, index_col=(0))
print(df)
print("Reading cstree")
tree = ct.df_to_cstree(df)

print("Sampling data")
data = tree.sample(n_samples)
print("Data sampled. Ensuring all columns have at least 2 unique values")
# make sure all columns are binary otherwise resample the data
while not all([data[col][1:].nunique() >= 2 for col in data.columns]):
    print("Resampling data until all columns have at least 2 unique values")
    data = tree.sample(n_samples)


# output
data.to_csv(snakemake.output[0], index=False)

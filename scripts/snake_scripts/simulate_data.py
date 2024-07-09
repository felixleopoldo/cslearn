from numpy import random as nprandom
import pandas as pd

from cstrees import cstree as ct

# input
input_path = snakemake.input[0]
n = int(snakemake.wildcards["n"])
seed = int(snakemake.wildcards["seed"])

# simulate data
nprandom.seed(seed)
df = pd.read_csv(input_path)
t = ct.df_to_cstree(df)
data = t.sample(n)

# output
data.to_csv(snakemake.output[0], index=False)

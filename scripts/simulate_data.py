from numpy import random as nprandom
import pandas as pd

from cstrees import cstree as ct

# input
input_path = path
n = n
seed = seed

# simulate data
nprandom.seed(seed)
df = pd.read_csv(input_path)
t = ct.df_to_cstree(df)
data = t.sample(n)

# output
output_path = f"data/n={n}/seed={seed}/" + input_path + "/data.csv"
df.to_csv(output_path)

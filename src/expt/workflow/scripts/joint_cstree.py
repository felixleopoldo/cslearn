import pandas as pd

from cstrees.cstree import df_to_cstree

# input
cstree_path = snakemake.input[0]
data_path = snakemake.input[1]

# load data
data = pd.read_csv(data_path)
# print("data:")
# print(data.head())
# load cstree and get joint distribution
cstree_df = pd.read_csv(cstree_path, index_col=(0))
# print("cstree_df:")
# print(cstree_df)
cstree = df_to_cstree(cstree_df)


dist_df = cstree.to_joint_distribution(
    label_order=list(data.columns), with_outcomes=False
)

# output
dist_df.to_csv(snakemake.output[0], quoting=0, index=False)

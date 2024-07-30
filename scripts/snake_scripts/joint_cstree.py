import pandas as pd

from cstrees.cstree import df_to_cstree

# input
cstree_path = snakemake.input[0]

# load cstree and get joint distribution
cstree_df = pd.read_csv(cstree_path, index_col=(0))
cstree = df_to_cstree(cstree_df)
dist_df = cstree.to_joint_distribution()

# output
dist_df.to_csv(snakemake.output[0])

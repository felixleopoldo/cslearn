from pandas import read_csv

from cstrees.cstree import df_to_cstree

# input
cstree_path = snakemake.input[0]

# load cstree and get joint distribution
df = read_csv(cstree_path, index_col=(0))
cstree = df_to_cstree(df)
# shuldnt we set the order also?
# we may hardcode it to 0,1,2,3,...
# It is like this by default, so not needed.

# the nodes till
dist_df = cstree.to_joint_distribution() 

# output
dist_df.to_csv(snakemake.output[0], index=False)

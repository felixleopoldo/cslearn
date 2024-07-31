import pandas as pd

from cstrees.cstree import df_to_cstree


# input
data_path = snakemake.input[1]
cstree_path = snakemake.input[0]

# estimate params
data = pd.read_csv(data_path)
cstree_df = pd.read_csv(cstree_path, index_col=(0))
cstree = df_to_cstree(cstree_df)

# alpha
# if snakemake.wildcards["cstree_param_est_estimation_type"]


cstree.estimate_stage_parameters(data)

cstree_df = cstree.to_df(write_probs=True)

# output
cstree_df.to_csv(snakemake.output["param_est"])

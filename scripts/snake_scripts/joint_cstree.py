from cstrees.cstree import df_to_cstree

# input
cstree = df_to_cstree(snakemake.input["input_cstree"])
data = snakemake.input["data"]

# estimate parameters and get joint distribution
cstree.estimate_stage_parameters(data)
dist_df = cstree.to_joint_distribution()

# output
df.to_csv(snakemake.output[0])

from os import path
import pandas as pd

# input
cslearn_pc_path = snakemake.input[0]
cslearn_grasp_path = snakemake.input[1]
# bos_path = snakemake.input[2]
# grasp_bhc_path = snakemake.input[3]
# p = int(snakemake.wildcards["cstree_p"])

# load times and aux
with open(cslearn_pc_path, "r") as f:
    pc_time = f.read()
with open(cslearn_grasp_path, "r") as f:
    grasp_time = f.read()

slashed = cslearn_pc_path.split("/")
p = slashed[3].split("=")[-1]
n = slashed[13].split("=")[-1]

# print(cslearn_pc_path)

# output
# if not path.exists(snakemake.output[0]):
#     with open(snakemake.output[0], "w") as f:
#         f.write("alg,p,n,time")

with open(snakemake.output[0], "w") as f:
    f.write(f"cslearn+pc,{p},{n},{pc_time}\n")
    f.write(f"cslearn+grasp,{p},{n},{grasp_time}\n")

# print(f"cslearn+pc,{p},{n},{pc_time}\n")

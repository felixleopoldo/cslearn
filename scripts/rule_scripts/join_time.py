from os import path
import pandas as pd

# input
cslearn_pc_paths = snakemake.input["cslearn_pc"]
cslearn_grasp_paths = snakemake.input["cslearn_grasp"]
# bos_path = snakemake.input[2]
# grasp_bhc_path = snakemake.input[3]

# load times and aux
algs_dict = {"cslearn+pc": cslearn_pc_paths, "cslearn+grasp": cslearn_grasp_paths}
joined_dict = {"alg": [], "p": [], "n": [], "time": []}
for alg, paths in algs_dict.items():
    for path in paths:
        with open(path, "r") as f:
            time = f.read()
        slashed = path.split("/")
        p = slashed[3].split("=")[-1]
        n = slashed[13].split("=")[-1]

        joined_dict["alg"] += [alg]
        joined_dict["p"] += [p]
        joined_dict["n"] += [n]
        joined_dict["time"] += [time]
times_df = pd.DataFrame(joined_dict)

# output
times_df.to_csv(snakemake.output[0], index=False)

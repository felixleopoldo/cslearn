from os import path
from itertools import chain

import pandas as pd

# input
csl_pc_paths = snakemake.input["cslearn_pc"]
csl_grasp_paths = snakemake.input["cslearn_grasp"]
bos_paths = snakemake.input["bos"]
# grasp_bhc_path = snakemake.input[3]

# load results
results = []
for path in chain(csl_pc_paths, csl_grasp_paths, bos_paths):
    results.append(pd.read_csv(path).to_dict())
times_df = pd.DataFrame(results)

# output
times_df.to_csv(snakemake.output[0], index=False)

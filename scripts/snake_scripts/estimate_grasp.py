from time import perf_counter as timer
import random
import warnings

from causallearn.search.PermutationBased.GRaSP import grasp
import pandas as pd

import cstrees.learning as ctl


warnings.simplefilter(action="ignore", category=FutureWarning)

# input
data_path = snakemake.input["data"]
seed = int(snakemake.wildcards["seed"])

# run grasp, get cvars, get dag
random.seed(seed)
data = pd.read_csv(data_path)

start = timer()
grasp_graph = grasp(data.values[1:], score_func="local_score_BDeu", maxP=10, depth=3)
end = timer()
runtime = end - start

dag_df = ctl.causallearn_graph_to_dag(grasp_graph, labels=data.columns, alg="grasp")

# output
# There should be a header. Otherwise we dont know which variable corresponds to which column.
dag_df.to_csv(snakemake.output["dag"], index=False, header=False) 

# This should be a csv file, see estimate_bos.R
with open(snakemake.output["time"], "w") as f:
    f.write(str(runtime))
    
# This should be a csv file, see estimate_bos.R
time_df = pd.DataFrame({"time": [runtime], "seed": [seed], 
                        "algorithm": ["grasp"], "p": [data.shape[1]], "n": [data.shape[0]]})

# write to csv
time_df.to_csv(snakemake.output["time_csv"], index=False)

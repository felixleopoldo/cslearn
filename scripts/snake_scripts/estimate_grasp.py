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
time_df = pd.DataFrame(
    {
        "method": ["GRaSP"],
        "time": [runtime],
        "seed": [seed],
        "p": [snakemake.wildcards["cstree_p"]],
        "n": [data.shape[0]],
    }
)

# output
dag_df.to_csv(snakemake.output["dag"], index=False, header=False)
time_df.to_csv(snakemake.output["time"], index=False)

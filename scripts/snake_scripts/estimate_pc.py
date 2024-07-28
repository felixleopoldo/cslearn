from time import perf_counter as timer

from causallearn.search.ConstraintBased.PC import pc
import pandas as pd

import cstrees.learning as ctl


# input
data_path = snakemake.input[0]

# run pc, get cvars, get dag
data = pd.read_csv(data_path, dtype=int)

start = timer()
pc_graph = pc(data.values[1:], 0.05, "gsq", node_names=data.columns)
end = timer()
runtime = end - start

dag_df = ctl.causallearn_graph_to_dag(pc_graph, labels=data.columns, alg="pc")
time_df = pd.DataFrame(
    {
        "method": ["PC"],
        "time": [runtime],
        "seed": [snakemake.wildcards["seed"]],
        "p": [snakemake.wildcards["cstree_p"]],
        "n": [data.shape[0]],
    }
)

# output
dag_df.to_csv(snakemake.output["dag"], index=False, header=False)
time_df.to_csv(snakemake.output["time"], index=False)

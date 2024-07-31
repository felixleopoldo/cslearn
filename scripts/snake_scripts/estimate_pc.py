from time import perf_counter as timer

from causallearn.search.ConstraintBased.PC import pc
import networkx as nx
import numpy as np
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

cpdag = nx.DiGraph()
cpdag.add_nodes_from(range(int(snakemake.wildcards["cstree_p"])))
incoming = np.argwhere(pc_graph.G.graph == -1)
cpdag.add_edges_from(incoming)

dag_df = ctl.causallearn_graph_to_dag(pc_graph, labels=data.columns, alg="pc")
cpdag_df = nx.to_pandas_adjacency(cpdag).astype(int)
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
dag_df.to_csv(snakemake.output["dag"], index=False)
cpdag_df.to_csv(snakemake.output["cpdag"], index=False)
time_df.to_csv(snakemake.output["time"], index=False)

from time import perf_counter as timer

from causaldag import PDAG
from causallearn.search.ConstraintBased.PC import pc
import pandas as pd
import numpy as np

import cslearn.learning as ctl
import networkx as nx

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

cpdag_adj = nx.to_numpy_array(cpdag)
nx_dag = PDAG.from_amat(cpdag_adj).to_dag().to_nx()
nx_dag.add_nodes_from(range(int(snakemake.wildcards["cstree_p"])))

dag_df = nx.to_pandas_adjacency(nx_dag).astype(int)
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

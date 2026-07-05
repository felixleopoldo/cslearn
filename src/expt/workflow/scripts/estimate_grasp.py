from time import perf_counter as timer
import random
import warnings

from causaldag import PDAG
from causallearn.search.PermutationBased.GRaSP import grasp
import networkx as nx
import pandas as pd
import numpy as np

import cslearn.learning as ctl


warnings.simplefilter(action="ignore", category=FutureWarning)

# input
data_path = snakemake.input[0]
seed = int(snakemake.wildcards["seed"])

# run grasp, get cvars, get dag
random.seed(seed)
data = pd.read_csv(data_path)

start = timer()
grasp_graph = grasp(data.values[1:], score_func="local_score_BDeu", maxP=10, depth=3)
end = timer()
runtime = end - start

cpdag = nx.DiGraph()
cpdag.add_nodes_from(range(int(snakemake.wildcards["cstree_p"])))
incoming = np.argwhere(grasp_graph.graph == -1)
cpdag.add_edges_from(incoming)

cpdag_adj = nx.to_numpy_array(cpdag)
nx_dag = PDAG.from_amat(cpdag_adj).to_dag().to_nx()
nx_dag.add_nodes_from(range(int(snakemake.wildcards["cstree_p"])))

dag_df = nx.to_pandas_adjacency(nx_dag).astype(int)
cpdag_df = nx.to_pandas_adjacency(cpdag).astype(int)
time_df = pd.DataFrame(
    {
        "method": ["GRaSP"],
        "time": [runtime],
        "total_time": [runtime],
        "seed": [seed],
        "p": [snakemake.wildcards["cstree_p"]],
        "n": [data.shape[0]],
    }
)

# output
dag_df.to_csv(snakemake.output["dag"], index=False)
cpdag_df.to_csv(snakemake.output["cpdag"], index=False)
time_df.to_csv(snakemake.output["time"], index=False)

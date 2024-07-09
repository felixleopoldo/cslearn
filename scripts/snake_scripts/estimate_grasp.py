import json
import random
import warnings

from causallearn.search.PermutationBased.GRaSP import grasp
import networkx as nx
import numpy as np
import pandas as pd
from pgmpy.base import PDAG

import cstrees.learning as ctl


warnings.simplefilter(action="ignore", category=FutureWarning)

# input
data_path = snakemake.input[0]
seed = int(snakemake.wildcards["seed"])
p = int(snakemake.wildcards["p"])

# run grasp, get cvars, get dag
random.seed(seed)
data = pd.read_csv(data_path)
grasp_graph = grasp(data.values[1:], score_func="local_score_BDeu", maxP=10, depth=3)

poss_cvars = ctl.causallearn_graph_to_posscvars(
    grasp_graph, labels=data.columns, alg="grasp"
)

adj = grasp_graph.graph
directed_mask = np.logical_and(adj == -1, adj.T == 1)
undirected_mask = np.logical_and(adj == -1, adj.T == -1)
directed_ebunch = [(u, v) for u, v in np.argwhere(directed_mask)]
undirected_ebunch = [(u, v) for u, v in np.argwhere(undirected_mask)]
cpdag = PDAG(directed_ebunch, undirected_ebunch)
dag = cpdag.to_dag()
nx_dag = nx.DiGraph(dag.edges())
nx_dag.add_nodes_from(range(p))
dag_df = nx.to_pandas_adjacency(nx_dag, dtype=int)

# output
dag_df.to_csv(snakemake.output[0], index=False, header=False)
with open(snakemake.output[1], "w") as f:
    json.dump(poss_cvars, f)

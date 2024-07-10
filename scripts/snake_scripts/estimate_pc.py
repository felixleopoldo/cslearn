import json
import random

from causallearn.search.ConstraintBased.PC import pc
import pandas as pd

import cstrees.learning as ctl


# input
data_path = snakemake.input[0]
p = int(snakemake.wildcards["p"])

# run pc, get cvars, get dag
data = pd.read_csv(data_path)
pc_graph = pc(data.values[1:], 0.05, "gsq", node_names=data.columns)

poss_cvars = ctl.causallearn_graph_to_posscvars(pc_graph, labels=data.columns, alg="pc")

dag_df = ctl.causallearn_graph_to_dag(pc_graph, labels=data.columns, alg="pc")

# output
dag_df.to_csv(snakemake.output[0], index=False, header=False)
with open(snakemake.output[1], "w") as f:
    json.dump(poss_cvars, f)

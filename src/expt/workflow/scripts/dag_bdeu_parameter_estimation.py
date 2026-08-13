import networkx as nx
import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork

# input
dag_df = pd.read_csv(snakemake.input[0], header=[0])

dag = nx.from_numpy_array(dag_df.values, create_using=nx.DiGraph)

# Relabel the nodes to strings (and the correct labels). Needed for pgmpy!
dag = nx.relabel_nodes(dag, {idx: str(label) for idx, label in enumerate(dag_df.columns)})

data = pd.read_csv(snakemake.input[1])

pgm = BayesianNetwork(dag)
pgm.fit(data[1:])  # row 0 contains cardinalities, not observations

pgm.save(snakemake.output[0])

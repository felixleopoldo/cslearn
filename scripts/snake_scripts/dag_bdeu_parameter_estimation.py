import networkx as nx
import pandas as pd
from pgmpy.models import BayesianNetwork

# input
dag = nx.from_pandas_adjacency(pd.read_csv(snakemake.input[0]))
data = pd.read_csv(snakemake.input[1])

# find MLE
pgm = BayesianNetwork(dag)
pgm.fit(data[1:])

# output
pgm.save(snakemake.output[0])

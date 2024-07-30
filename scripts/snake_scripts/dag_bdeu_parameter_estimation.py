import networkx as nx
import pandas as pd
from pgmpy.models import BayesianNetwork

# input

dag_adjmat = pd.read_csv(snakemake.input[0], header=None)
print("dag_adjmat:")
print(dag_adjmat)
print(dag_adjmat.values)

print("read dag")
dag = nx.from_numpy_array(dag_adjmat.values, create_using=nx.DiGraph)

# print edges
print("edges:")
print(dag.edges)

print("dag:")
print(dag)

data = pd.read_csv(snakemake.input[1])
# find MLE

print("create pgmpy bn")
pgm = BayesianNetwork(dag)
pgm.fit(data[1:])

# output
pgm.save(snakemake.output[0])

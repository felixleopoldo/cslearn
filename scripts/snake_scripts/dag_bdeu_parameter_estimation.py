import networkx as nx
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator

# input
dag_df = pd.read_csv(snakemake.input[0], header=[0])
dag = nx.from_numpy_array(dag_df.values, create_using=nx.DiGraph)

print(f"\nis a DAG: {nx.is_directed_acyclic_graph(dag)}; has adj_mat:\n{dag_df}\n")

# Relabel the nodes to strings (and the correct labels). Needed for pgmpy!
dag = nx.relabel_nodes(
    dag, {idx: str(label) for idx, label in enumerate(dag_df.columns)}
)

data = pd.read_csv(snakemake.input[1])
# find MLE

pgm = BayesianNetwork(dag)
# print("fit pgmpy bn")
# print(pgm)
# print(pgm.edges)

# using https://pgmpy.org/param_estimator/mle.html
# https://pgmpy.org/models/bayesiannetwork.html#module-pgmpy.models.BayesianNetwork

# state_names = {col: [0, 1] for col in data.columns}
# estimator = MaximumLikelihoodEstimator(pgm, data[1:], state_names=state_names)


# estimator.estimate_cpd(0)
pgm.fit(data[1:])
# print(pgm.get_cpds())

pgm.save(snakemake.output[0])

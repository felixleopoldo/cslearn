import networkx as nx
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator

# input

dag_adjmat = pd.read_csv(snakemake.input[0], header=None)
dag = nx.from_numpy_array(dag_adjmat.values, create_using=nx.DiGraph)

# Relabel the nodes to strings. Needed for pgmpy!
dag = nx.relabel_nodes(dag, {i: str(i) for i in range(dag_adjmat.shape[0])})

data = pd.read_csv(snakemake.input[1])
# find MLE


pgm = BayesianNetwork(dag)
print("fit pgmpy bn")
print(pgm)
print(pgm.edges)

# using https://pgmpy.org/param_estimator/mle.html
# https://pgmpy.org/models/bayesiannetwork.html#module-pgmpy.models.BayesianNetwork

state_names = {col: [0, 1] for col in data.columns}
#estimator = MaximumLikelihoodEstimator(pgm, data[1:], state_names=state_names)


#estimator.estimate_cpd(0)
pgm.fit(data[1:])
#print(pgm.get_cpds())

pgm.save(snakemake.output[0])

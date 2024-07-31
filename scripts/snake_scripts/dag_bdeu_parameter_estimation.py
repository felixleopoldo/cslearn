import networkx as nx
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator

# input

dag_adjmat = pd.read_csv(snakemake.input[0], header=None)
print("dag_adjmat:")
print(dag_adjmat)
print(dag_adjmat.values)

print("read dag")
dag = nx.from_numpy_array(dag_adjmat.values, create_using=nx.DiGraph)

# Relabel the nodes to strings. Needed for pgmpy!
dag = nx.relabel_nodes(dag, {i: str(i) for i in range(dag_adjmat.shape[0])})

# print edges
print("edges:")
print(dag.edges)

print("dag:")
print(dag)

data = pd.read_csv(snakemake.input[1])
# find MLE

print("create pgmpy bn")
pgm = BayesianNetwork(dag)
print("fit pgmpy bn")
print(pgm)
print(pgm.edges)
print("fit parameters")
print(data[1:])

# using https://pgmpy.org/param_estimator/mle.html
# https://pgmpy.org/models/bayesiannetwork.html#module-pgmpy.models.BayesianNetwork
# an alternative would be causalnex
# https://github.com/felixleopoldo/cancer_challenge/blob/814c03a210d4533062915e656ae92284db81acc9/challenge2/tozip/code/docalc.ipynb

state_names = {col: [0, 1] for col in data.columns}
#estimator = MaximumLikelihoodEstimator(pgm, data[1:], state_names=state_names)


#estimator.estimate_cpd(0)
pgm.fit(data[1:])
print(pgm.get_cpds())
print("check")
print(pgm.check_model())
#estimator.get_parameters()
# output
pgm.save(snakemake.output[0])

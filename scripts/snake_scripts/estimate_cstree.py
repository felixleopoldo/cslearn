from time import perf_counter as timer
import json
import warnings


from pandas import read_csv

import cstrees.learning as ctl
import cstrees.scoring as sc


warnings.simplefilter(action="ignore", category=FutureWarning)

# input
data_path = snakemake.input[0]
poss_cvars_path = snakemake.input[1]
prev_runtime_path = snakemake.input[2]
max_cvars = int(snakemake.wildcards["cslearn_max_cvar"])
num_iter = int(snakemake.wildcards["cslearn_mcmc_iterations"])
alpha_tot = float(snakemake.wildcards["cslearn_alpha_tot"])
prior = snakemake.wildcards["cslearn_param_prior"]

# load inputs
data = read_csv(data_path)
with open(poss_cvars_path, "r") as f:
    poss_cvars = json.load(f)
if poss_cvars == "":
    poss_cvars = None

# estimate cstree
start = timer()
score_table, context_scores, _ = sc.order_score_tables(
    data, max_cvars=max_cvars, alpha_tot=alpha_tot, method=prior, poss_cvars=poss_cvars
)
orders, scores = ctl.gibbs_order_sampler(num_iter, score_table)
map_order = orders[scores.index(max(scores))]
opt_tree = ctl._optimal_cstree_given_order(map_order, context_scores)
end = timer()
runtime = end - start

with open(prev_runtime_path, "r") as f:
    prev_runtime = f.read()
total_runtime = float(prev_runtime) + runtime

cstree_df = opt_tree.to_df()

# output
cstree_df.to_csv(snakemake.output[0])
with open(snakemake.output[1], "w") as f:
    f.write(str(runtime))
with open(snakemake.output[2], "w") as f:
    f.write(str(total_runtime))

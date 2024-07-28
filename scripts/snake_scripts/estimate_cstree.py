from time import perf_counter as timer
import json
import warnings

import pandas as pd

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
data = pd.read_csv(data_path)
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

cvar_alg_df = pd.read_csv(prev_runtime_path)
cvar_alg = cvar_alg_df["method"]
cvar_runtime = cvar_alg_df["time"]
total_runtime = float(cvar_runtime) + runtime

cstree_df = opt_tree.to_df()
time_df = pd.DataFrame(
    {
        "method": ["CSlearn+" + cvar_alg],
        "time": [cvar_runtime],
        "total_time": [runtime],
        "seed": [snakemake.wildcards["seed"]],
        "p": [snakemake.wildcards["p"]],
        "n": [snakemake.wildcards["n"]],
    }
)

# output
cstree_df.to_csv(snakemake.output["cstree"])
time_df.to_csv(snakemake.output["time"], index=False)

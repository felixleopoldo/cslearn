import cstrees.learning as ctl
import cstrees.scoring as sc


# input
data = snakemake.input["data"]
poss_cvars = snakemake.input["poss_cvars"]

# estimate cstree
score_table, context_scores, _ = sc.order_score_tables(
    data, max_cvars=2, alpha_tot=1.0, method="BDeu", poss_cvars=poss_cvars
)

# run Gibbs sampler to get MAP order
orders, scores = ctl.gibbs_order_sampler(5000, score_table)
map_order = orders[scores.index(max(scores))]

# estimate CStree
opt_tree = ctl._optimal_cstree_given_order(map_order, context_scores)
cstree_df = opt_tree.to_df(write_probs=True)

# output
cstree_df.to_csv(snakemake.output[0])

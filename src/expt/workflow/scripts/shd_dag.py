import pandas as pd
import cstrees.cstree as ct


def shd_ldag(est_edges, true):
    true_edges = set(true.to_LDAG().edges())
    est_set    = set(est_edges)
    reversals  = sum(1 for u, v in est_set if (v, u) in true_edges)
    return len(est_set.symmetric_difference(true_edges)) - reversals


true_tree = ct.df_to_cstree(pd.read_csv(snakemake.input["true"], index_col=0))

cpdag = pd.read_csv(snakemake.input["est"], index_col=0)
labels = [str(c) for c in cpdag.columns]
adj = cpdag.values.astype(int)
est_edges = [
    (labels[i], labels[j])
    for i in range(len(labels))
    for j in range(len(labels))
    if adj[i, j] == 1
]

pd.DataFrame({
    "method": [snakemake.params["method"]],
    "shd":    [shd_ldag(est_edges, true_tree)],
    "seed":   [snakemake.wildcards["seed"]],
    "p":      [snakemake.wildcards["cstree_p"]],
    "n":      [snakemake.wildcards["cstree_data_n"]],
}).to_csv(snakemake.output[0], index=False)

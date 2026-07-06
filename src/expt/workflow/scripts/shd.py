import pandas as pd

import cslearn.cstree as ct


def shd_ldag(estimated, true):
    est_edges = set(estimated.to_LDAG().edges())
    true_edges = set(true.to_LDAG().edges())
    reversals = sum(1 for u, v in est_edges if (v, u) in true_edges)
    return len(est_edges.symmetric_difference(true_edges)) - reversals


true_tree = ct.df_to_cstree(pd.read_csv(snakemake.input["true"], index_col=0))
est_tree = ct.df_to_cstree(pd.read_csv(snakemake.input["est"], index_col=0))

phase1_alg = snakemake.wildcards["cslearn_poss_cvars"].split("/")[0].upper()
method = f"CSlearn+{phase1_alg}"

pd.DataFrame(
    {
        "method": [method],
        "shd": [shd_ldag(est_tree, true_tree)],
        "seed": [snakemake.wildcards["seed"]],
        "p": [snakemake.wildcards["cstree_p"]],
        "n": [snakemake.wildcards["cstree_data_n"]],
    }
).to_csv(snakemake.output[0], index=False)

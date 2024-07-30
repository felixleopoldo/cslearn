import pandas as pd

# input
true_path = snakemake.input[0]
est_path = snakemake.input[1]
alg = snakemake.wildcards["alg"]
# print(alg)

# comput kl-div
# see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html#scipy.stats.entropy
# this implementation is like the scipy one but directly uses the
# recorded log probabalities
true = pd.read_csv(true_path, index_col=[0])
est = pd.read_csv(est_path, index_col=[0])

kl_div = (true["prob"] * (true["log_prob"] - est["log_prob"])).sum()

kl_df = pd.DataFrame(
    {
        "method": ["test_method"],
        "kl_div": [kl_div],
        "seed": [snakemake.wildcards["seed"]],
        "p": [snakemake.wildcards["cstree_p"]],
        "n": ["test_n"],
    }
)

# output
kl_df.to_csv(snakemake.output[0], index=False)

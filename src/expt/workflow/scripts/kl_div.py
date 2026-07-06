import numpy as np
import pandas as pd

# input
true_path = snakemake.input[0]
est_path = snakemake.input[1]

n = int(snakemake.wildcards["data"].split("/")[1].split("=")[1])
alg = snakemake.wildcards["alg"].split("/")[0]
if alg == "cslearn":
    alg += "+" + snakemake.wildcards["alg"].split("/")[2].split("=")[1]
param_method = snakemake.wildcards["param_est"].split("/")[1].split("=")[1]

# comput kl-div
# see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html#scipy.stats.entropy
# this implementation is like the scipy one but directly uses the
# recorded log probabalities
true = pd.read_csv(true_path)
est = pd.read_csv(est_path)

est.loc[est["prob"] == 0, "log_prob"] = np.log(1e-5)

unsummed_kl_div = true["prob"] * (true["log_prob"] - est["log_prob"])
unsummed_kl_div[true["prob"] == 0] = 0
kl_div = unsummed_kl_div.sum()
if kl_div > 10:
    print(true["prob"], est["prob"])

kl_df = pd.DataFrame(
    {
        "method": [f"{alg} ({param_method})"],
        "kl_div": [kl_div],
        "seed": [snakemake.wildcards["seed"]],
        "p": [snakemake.wildcards["cstree_p"]],
        "n": [n],
    }
)

# output
kl_df.to_csv(snakemake.output[0], index=False)

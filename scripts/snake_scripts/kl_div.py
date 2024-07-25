import pandas as pd

# input
true_path = snakemake.input["true"]
est_path = snakemake.input["est"]

# comput kl-div
# see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html#scipy.stats.entropy
# this implementation is like the scipy one but directly uses the
# recorded log probabalities
true = pd.read_csv(true_path, index_col=[0])
est = pd.read_csv(est_path, index_col=[0])

kl_div = (true["prob"] * (true["log_prob"] - est["log_prob"])).sum()
print(kl_div)
# output

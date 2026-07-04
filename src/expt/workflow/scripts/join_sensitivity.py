import pandas as pd

# Map named input groups to their true β (max_cvar) values.
group_beta = {
    "correct": 2,  # β_true=2, β_est=2 (correctly specified)
    "under":   3,  # β_true=3, β_est=2 (under-specified)
    "over":    1,  # β_true=1, β_est=2 (over-specified)
}

results = []
for key, files in snakemake.input.items():
    true_beta = group_beta[key]
    for f in files:
        df = pd.read_csv(f)
        df["true_max_cvar"] = true_beta
        results.append(df)

pd.concat(results).to_csv(snakemake.output[0], index=False)

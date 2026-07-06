import pandas as pd

# join results
results = []
for key, value in snakemake.input.items():
    for fpath in value:
        results.append(pd.read_csv(fpath))

times_df = pd.concat(results)

# output
times_df.to_csv(snakemake.output[0], index=False)

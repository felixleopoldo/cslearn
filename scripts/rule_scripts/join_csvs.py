from os import path
from itertools import chain

import pandas as pd

# join results
results = []
for key, value in snakemake.input.items():
        for path in value:
            results.append(pd.read_csv(path))

times_df = pd.concat(results)

# output
times_df.to_csv(snakemake.output[0], index=False)

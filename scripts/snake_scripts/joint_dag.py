from itertools import product

import numpy as np
import pandas as pd
from pgmpy.models import BayesianNetwork
from tqdm import tqdm


# input
pgm = BayesianNetwork.load(snakemake.input[0])

cards = pgm.number_of_nodes() * [2]

# Iterate over all possible outcomes and calculate the probability mass function.
# Store the outcomes together with the probabilities in a Pandas Dataframe.
outcomes = product(*[range(card) for card in cards])
n_outcomes = np.prod(cards)

# Create an empty dataframe with the correct column names
# df_outcomes = pd.DataFrame(columns=self.labels)
df_outcomes = pd.DataFrame(columns=list(pgm))
# store all the outcomes and probabilities
pmfs = [None] * np.prod(cards)

for i, outcome in tqdm(
    enumerate(outcomes), total=n_outcomes, desc="Calculating joint distribution"
):
    df_outcomes.loc[i] = outcome
    pgmpy_outcome = {var: marg_outcome for var, marg_outcome in zip(list(pgm), outcome)}
    pmfs[i] = pgm.get_state_probability(pgmpy_outcome)

    # logprob not given by pgmpy, but could implement based on
    # get_state_probabality() source if needed

df_pmf = pd.DataFrame(pmfs, columns=["prob"])
df_pmf_log = pd.DataFrame(np.log(pfms), columns=["log_prob"])
# join the two dataframes
df = pd.concat([df_outcomes, df_pmf, df_pmf_log], axis=1)

# output
df.to_csv(snakemake.output[0])

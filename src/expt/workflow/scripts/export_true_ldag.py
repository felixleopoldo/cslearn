import pandas as pd

import cslearn.cstree as ct

tree = ct.df_to_cstree(pd.read_csv(snakemake.input[0], index_col=0))
edges = pd.DataFrame(list(tree.to_LDAG().edges()), columns=["u", "v"])
edges.to_csv(snakemake.output[0], index=False)

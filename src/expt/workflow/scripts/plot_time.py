import pandas as pd
import seaborn as sns
from labels import canonical_method_label
from palette import HATCHES, PALETTE

plot_data = pd.read_csv(snakemake.input[0])
plot_data["method"] = plot_data["method"].map(canonical_method_label)
plot_data["hue"] = plot_data["method"] + ", n=" + plot_data["n"].astype(str)

sns.set(font_scale=1.25)
sns.set_style("white")
sns.set_style({"legend.frameon": False})
g = sns.boxplot(data=plot_data, x="p", y="total_time", hue="hue", palette=PALETTE)

# Grayscale-safe hatching. Patches are ordered hue-first (all x-positions for
# hue 0, then hue 1, ...), so hue index = i // n_x.
_n = plot_data["hue"].nunique()
_nx = plot_data["p"].nunique()
for i, patch in enumerate(g.patches[: _n * _nx]):
    patch.set_hatch(HATCHES[(i // _nx) % len(HATCHES)])
for i, handle in enumerate(g.legend_.legend_handles):
    handle.set_hatch(HATCHES[i % len(HATCHES)])

g.set(title="", xlabel="number of variables ($p$)", ylabel="runtime (seconds)")
g.legend_.set_title("")
g.legend_.set_frame_on(False)
g.figure.tight_layout()
g.figure.savefig(snakemake.output[0])

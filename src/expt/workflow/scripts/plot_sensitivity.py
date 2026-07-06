import pandas as pd
import seaborn as sns
from palette import HATCHES, PALETTE

plot_data = pd.read_csv(snakemake.input[0])

label_map = {
    1: "over-spec ($\\beta_{\\mathrm{true}}$=1)",
    2: "correct ($\\beta_{\\mathrm{true}}$=2)",
    3: "under-spec ($\\beta_{\\mathrm{true}}$=3)",
}
plot_data["hue"] = plot_data["true_max_cvar"].map(label_map) + ", n=" + plot_data["n"].astype(str)

# Order hues: over-spec < correct < under-spec, within each by n ascending.
hue_order = [f"{label_map[b]}, n={n}" for b in [1, 2, 3] for n in sorted(plot_data["n"].unique())]
hue_order = [h for h in hue_order if h in plot_data["hue"].unique()]

sns.set(font_scale=1.25)
sns.set_style("white")
sns.set_style({"legend.frameon": False})
g = sns.boxplot(data=plot_data, x="p", y="shd", hue="hue", hue_order=hue_order, palette=PALETTE)

# Grayscale-safe hatching. Patches are ordered hue-first (all x-positions for
# hue 0, then hue 1, ...), so hue index = i // n_x.
_n = plot_data["hue"].nunique()
_nx = plot_data["p"].nunique()
for i, patch in enumerate(g.patches[: _n * _nx]):
    patch.set_hatch(HATCHES[(i // _nx) % len(HATCHES)])
for i, handle in enumerate(g.legend_.legend_handles):
    handle.set_hatch(HATCHES[i % len(HATCHES)])

g.set(title="", xlabel="number of variables ($p$)", ylabel="SHD (LDAG)")
g.legend_.set_title("")
g.legend_.set_frame_on(False)
g.figure.tight_layout()
g.figure.savefig(snakemake.output[0])

import pandas as pd
import seaborn as sns
from labels import canonical_method_label
from palette import method_hue_style, sorted_hues, style_figure

plot_data = pd.read_csv(snakemake.input[0])
plot_data["method"] = plot_data["method"].map(canonical_method_label)
plot_data["hue"] = plot_data["method"] + ", n=" + plot_data["n"].astype(str)

hue_order = sorted_hues(plot_data["hue"].unique())
palette, hatches = method_hue_style(hue_order)

sns.set(font_scale=1.6)
sns.set_style("white")
sns.set_style({"legend.frameon": False})
g = sns.boxplot(data=plot_data, x="p", y="shd", hue="hue", hue_order=hue_order, palette=palette, linewidth=1.2)

_n = len(hue_order)
_nx = plot_data["p"].nunique()
for i, patch in enumerate(g.patches[: _n * _nx]):
    patch.set_hatch(hatches[hue_order[i // _nx]])
for handle, hue in zip(g.legend_.legend_handles, hue_order):
    handle.set_hatch(hatches[hue])

g.set(title="", xlabel="number of variables ($p$)", ylabel="SHD (LDAG)")
style_figure(g, n_x=_nx, n_hue=_n)
g.figure.savefig(snakemake.output[0], bbox_inches="tight")

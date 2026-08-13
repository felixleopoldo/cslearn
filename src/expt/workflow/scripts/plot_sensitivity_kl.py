import pandas as pd
import seaborn as sns
from palette import beta_hue_style, style_figure

plot_data = pd.read_csv(snakemake.input[0])

label_map = {
    1: "over-spec ($\\beta_{\\mathrm{true}}$=1)",
    2: "correct ($\\beta_{\\mathrm{true}}$=2)",
    3: "under-spec ($\\beta_{\\mathrm{true}}$=3)",
}
plot_data["hue"] = plot_data["true_max_cvar"].map(label_map) + ", n=" + plot_data["n"].astype(str)

ns = sorted(plot_data["n"].unique())
hue_order = [f"{label_map[b]}, n={n}" for b in [1, 2, 3] for n in ns]
hue_order = [h for h in hue_order if h in plot_data["hue"].unique()]
palette = beta_hue_style(label_map, ns)

sns.set(font_scale=0.9)
sns.set_style("white")
sns.set_style({"legend.frameon": False})
g = sns.boxplot(data=plot_data, x="p", y="kl_div", hue="hue", hue_order=hue_order, palette=palette, linewidth=1.0)

g.set(title="", xlabel="number of variables ($p$)", ylabel="KL divergence")
style_figure(g, n_hue=len(hue_order))
g.figure.savefig(snakemake.output[0], bbox_inches="tight")

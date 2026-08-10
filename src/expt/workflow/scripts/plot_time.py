import pandas as pd
import seaborn as sns
from labels import baseline_display_label, canonical_method_label
from palette import method_hue_style, sorted_hues, style_figure

plot_data = pd.read_csv(snakemake.input[0])
plot_data["method"] = plot_data["method"].map(canonical_method_label)
plot_data["hue"] = plot_data["method"] + ", n=" + plot_data["n"].astype(str)

hue_order = sorted_hues(plot_data["hue"].unique())
palette = method_hue_style(hue_order)

sns.set(font_scale=0.9)
sns.set_style("white")
sns.set_style({"legend.frameon": False})
g = sns.boxplot(
    data=plot_data, x="p", y="total_time", hue="hue", hue_order=hue_order, palette=palette, linewidth=1.0
)

g.set(title="", xlabel="number of variables ($p$)", ylabel="runtime (seconds)")
style_figure(g, n_hue=len(hue_order))
for text, hue in zip(g.legend_.get_texts(), hue_order):
    text.set_text(baseline_display_label(hue))
g.figure.savefig(snakemake.output[0], bbox_inches="tight")

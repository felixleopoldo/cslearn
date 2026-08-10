import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from labels import BASELINE_METHODS, canonical_method_label
from palette import method_hue_style, sorted_hues

plot_data = pd.read_csv(snakemake.input[0])
plot_data["method"] = plot_data["method"].map(canonical_method_label)

_SUFFIX_RE = re.compile(r"^(?P<base>.+) \((?P<suffix>MLE|MAP)\)$")


def split_suffix(label):
    m = _SUFFIX_RE.match(label)
    return (m["base"], m["suffix"]) if m else (label, None)


plot_data["_base_method"], plot_data["_param"] = zip(*plot_data["method"].map(split_suffix))
plot_data["hue"] = plot_data["_base_method"] + ", n=" + plot_data["n"].astype(str)

hue_order = sorted_hues(plot_data["hue"].unique())
palette = method_hue_style(hue_order)

# Facet MLE/MAP into stacked rows (each gets the full print width) instead of
# cramming a 3rd dimension (method x n x MLE/MAP) into one panel via
# color+hatch, per palette.py's rationale. Falls back to a single panel when
# a CSV has no MAP data (e.g. kl_divergence_2c).
param_order = [p for p in ["MLE", "MAP"] if p in plot_data["_param"].unique()] or [None]

sns.set(font_scale=0.9)
sns.set_style("white")
sns.set_style({"legend.frameon": False})

legend_height = 0.16 * len(hue_order) + 0.15
fig, axes = plt.subplots(
    len(param_order), 1, figsize=(2.9, 1.9 * len(param_order) + legend_height), sharex=True, sharey=True
)
axes = [axes] if len(param_order) == 1 else list(axes)

# Only meaningful when the CSV actually has a real MLE-vs-MAP split for some
# method (kl_divergence_2a/2b): methods with no MAP variant (the plain
# PC/GRaSP baselines, only ever estimated via MLE) act as a fixed reference
# and should show up in every row, not get siloed into the MLE facet. When
# nothing in the CSV has a MAP variant at all (kl_divergence_2c -- every row
# is "(MLE)", but that's not an MLE-vs-MAP comparison, just a flat
# method-vs-method one, like time_3a/shd_c), there's only one row and this
# doesn't apply.
has_map_facet = "MAP" in plot_data["_param"].unique()
methods_with_map = set(plot_data.loc[plot_data["_param"] == "MAP", "_base_method"].unique()) if has_map_facet else set()

# The row title names the CSlearn method whose estimator the row's title
# actually describes -- the baseline (no MAP variant) isn't "MAP" just
# because it's shown in that row too; see its own legend label below.
cslearn_method = "/".join(sorted(methods_with_map))

for ax, param in zip(axes, param_order):
    if param is None:
        subset = plot_data
    else:
        subset = plot_data[(plot_data["_param"] == param) | (~plot_data["_base_method"].isin(methods_with_map))]
    sns.boxplot(data=subset, x="p", y="kl_div", hue="hue", hue_order=hue_order, palette=palette, linewidth=1.0, ax=ax)
    ax.get_legend().remove()
    ax.set(xlabel="", ylabel="KL-divergence")
    if has_map_facet:
        ax.set_title(f"{cslearn_method} ({param})", loc="left", fontsize="small")

axes[-1].set_xlabel("number of variables ($p$)")


def display_label(hue):
    """"Baseline" (PC/GRaSP, no CSlearn refinement) is a property of the
    method itself -- see BASELINE_METHODS -- so it's labeled consistently
    here and in every other figure that shows it (plot_time.py, plot_shd.py),
    regardless of whether this particular CSV has a real MLE-vs-MAP split.
    When it does (kl_divergence_2a/2b), the baseline is shown unchanged in
    every row, so its label also states its estimator explicitly -- otherwise
    it reads as "we also computed a MAP version of the baseline", which is
    false."""
    method, n = hue.rsplit(", n=", 1)
    if method not in BASELINE_METHODS:
        return hue
    suffix = " (MLE)" if has_map_facet else ""
    return f"{method} baseline{suffix}, n={n}"


handles = [plt.Rectangle((0, 0), 1, 1, facecolor=palette[h], edgecolor="black") for h in hue_order]
fig.legend(
    handles,
    [display_label(h) for h in hue_order],
    loc="upper center",
    bbox_to_anchor=(0.5, -0.22),
    bbox_transform=axes[-1].transAxes,
    ncol=1,
    frameon=False,
)

fig.savefig(snakemake.output[0], bbox_inches="tight")

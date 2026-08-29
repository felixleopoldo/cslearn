import matplotlib.patches as mpatches
import pandas as pd
from palette import (
    LEGEND_PAD_IN,
    LEGEND_ROW_H_IN,
    METHOD_COLORS,
    grow_to_fit,
    reserve_legend_margin,
    set_plot_style,
    square_grid,
)

# CSlearn+PC only, matching the supplement's existing "Scalability Analysis"
# discussion (which is PC-specific) -- not a general method comparison, so
# this doesn't reuse METHOD_MARKERS/add_shared_legend's per-method machinery.
df = pd.read_csv(snakemake.input[0])
df = df[(df["method"] == "CSlearn+PC") & df["n"].isin([1000, 1001])].copy()
# n is 1000 or 1001 depending on the run (an off-by-one in data generation,
# not a different experimental condition) -- both are "n=1000" for this plot.

# total_time = phase-1 (constraint-based/PC) time + phase-2/3 (Gibbs sampler
# + exact search) time (see estimate_cstree.py) -- recover the phase-1
# component by subtraction, since only the phase-2/3 component and the
# total are stored directly.
df["constraint_phase"] = df["total_time"] - df["time"]
df["search_phase"] = df["time"]

medians = df.groupby("p")[["constraint_phase", "search_phase"]].median().sort_index()

# Reuse Figure 4's PC/CSlearn+PC colors: the constraint-based phase is a PC
# run, and the search phase is what CSlearn spends beyond that.
CONSTRAINT_COLOR = METHOD_COLORS["PC"]
SEARCH_COLOR = METHOD_COLORS["CSlearn+PC"]
# Hatching, not just color, distinguishes the two bar segments in black and
# white (JCGS's print requirement -- see palette.py's module docstring on
# why every figure here needs a non-color channel).
CONSTRAINT_HATCH = ""
SEARCH_HATCH = "///"

set_plot_style()
fig, axes = square_grid(1, 1, panel_size=2.6)
ax = axes[0]

x = range(len(medians))
ax.bar(
    x,
    medians["constraint_phase"],
    color=CONSTRAINT_COLOR,
    hatch=CONSTRAINT_HATCH,
    edgecolor="black",
    linewidth=0.6,
)
ax.bar(
    x,
    medians["search_phase"],
    bottom=medians["constraint_phase"],
    color=SEARCH_COLOR,
    hatch=SEARCH_HATCH,
    edgecolor="black",
    linewidth=0.6,
)
ax.set_xticks(list(x))
ax.set_xticklabels([str(p) for p in medians.index])
ax.set_xlabel("number of variables ($p$)")
ax.set_ylabel("time (seconds)")

# Bespoke Patch-based legend, not add_shared_legend, which draws line+marker
# handles for method/n comparisons -- a mismatch for two stacked bar
# segments. Still uses grow_to_fit/reserve_legend_margin for a reserved band.
handles = [
    mpatches.Patch(
        facecolor=CONSTRAINT_COLOR, hatch=CONSTRAINT_HATCH, edgecolor="black", linewidth=0.6, label="constraint-based phase (PC)"
    ),
    mpatches.Patch(
        facecolor=SEARCH_COLOR, hatch=SEARCH_HATCH, edgecolor="black", linewidth=0.6, label="Gibbs sampler and exact search"
    ),
]

grow_to_fit(fig, ax)

# ncol=1: both labels are long enough that fitting them side by side would
# need more width than this single narrow panel has -- stacking them as two
# rows (reserving 2 * LEGEND_ROW_H_IN) is what actually fits without
# truncating either label.
legend_h = 2 * LEGEND_ROW_H_IN + LEGEND_PAD_IN
frac = legend_h / (fig.get_figheight() + legend_h)
reserve_legend_margin(fig, "above", frac)

fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), frameon=False, ncol=1)

fig.savefig(snakemake.output[0])

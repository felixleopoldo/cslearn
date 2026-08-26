import pandas as pd
from palette import (
    BETA_COLORS,
    BETA_LABELS,
    BETA_MARKERS,
    add_shared_legend,
    draw_lines,
    grow_to_fit,
    linestyles_for,
    reserve_legend_margin,
    set_plot_style,
    square_grid,
)

kl_data = pd.read_csv(snakemake.input.kl)
shd_data = pd.read_csv(snakemake.input.shd)
kl_data["n"] = kl_data["n"].astype(int)
shd_data["n"] = shd_data["n"].astype(int)

all_betas = sorted(set(kl_data["true_max_cvar"].unique()) | set(shd_data["true_max_cvar"].unique()))
all_ns = sorted(set(kl_data["n"].unique()) | set(shd_data["n"].unique()))
linestyle_map = linestyles_for(all_ns)

set_plot_style()
# KL divergence and SHD are different metrics (different y-axes), so panels
# don't share a y-axis the way every other combined figure's panels do --
# sharey=False keeps each panel's own y-tick labels instead of incorrectly
# treating one as redundant with the other.
fig, axes = square_grid(1, 2, panel_size=2.1, sharey=False)

draw_lines(axes[0], kl_data, "p", "kl_div", "true_max_cvar", "n", BETA_COLORS, BETA_MARKERS, linestyle_map)
axes[0].set_yscale("log")
axes[0].set_ylabel("KL divergence")
# Titles name what each metric means here, not just repeat the y-axis label:
# the two panels differ only by metric (same experiment, same beta-arm/n
# legend), so the y-label already states *which* metric -- the title adds
# *why it's the one shown here* (distributional vs. structural accuracy).
axes[0].set_title("(a) distributional accuracy", loc="left")

draw_lines(axes[1], shd_data, "p", "shd", "true_max_cvar", "n", BETA_COLORS, BETA_MARKERS, linestyle_map)
axes[1].set_ylabel("SHD (LDAG)")
axes[1].set_title("(b) structural accuracy", loc="left")

fig.supxlabel("number of variables ($p$)")

# See plot_time.py's comment: growing height only (grow_to_fit) doesn't
# affect a "beside" legend's width-fraction, so frac doesn't need
# recomputing after growth here. Both panels show their own y-tick labels
# (sharey=False), so the reference axes for growth doesn't matter -- either
# panel's box grows identically to the other's.
grow_to_fit(fig, axes[0])
FRAC = 0.22
reserve_legend_margin(fig, "beside", FRAC)

hue_entries = [(BETA_LABELS[b], BETA_COLORS[b], BETA_MARKERS[b]) for b in all_betas]
style_entries = [(f"n={n}", linestyle_map[n]) for n in all_ns]
add_shared_legend(fig, hue_entries, style_entries, loc="beside", frac=FRAC, hue_ncol=1)

fig.savefig(snakemake.output[0])

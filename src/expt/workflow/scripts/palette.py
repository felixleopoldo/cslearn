"""Shared color/marker/linestyle logic and line-drawing for the experiment figures.

JCGS prints in black and white, so each figure's two comparison dimensions
(method or beta-arm, and n) each get a non-color channel: method is color +
marker shape, n is linestyle. Color alone isn't a safe channel for either.
"""

import math
import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from labels import canonical_method_label

# Generated offline with qualpal (github.com/jolars/qualpal): a
# farthest-point search over CIEDE2000 distance in LCh(ab) space for colors
# that stay mutually distinct under simulated color-vision deficiency and
# against this project's plot background (seaborn darkgrid's #EAEAF2),
# capped at l=0.1-0.4 to avoid washing out against that background. Fixed
# colors, hardcoded here rather than regenerated at runtime.
METHOD_COLORS = {
    "CSlearn+PC": "#b8420a",
    "PC": "#0caebd",
    "CSlearn+GRaSP": "#394d04",
    "GRaSP": "#acae1c",
    "BOS": "#b20aa6",
    "GRaSP+BHC": "#051b31",
}

METHOD_MARKERS = {
    "CSlearn+PC": "o",
    "PC": "s",
    "CSlearn+GRaSP": "^",
    "GRaSP": "D",
    "BOS": "v",
    "GRaSP+BHC": "P",
}

# Base colors/markers for the sensitivity plots' beta-misspecification arms.
# Also qualpal-generated (see METHOD_COLORS above; same params, generate(3)).
BETA_COLORS = {
    1: "#a6108e",  # over-spec
    2: "#0a2a19",  # correct
    3: "#acae1c",  # under-spec
}

BETA_MARKERS = {
    1: "^",
    2: "o",
    3: "v",
}

# Sensitivity legend/plot labels -- just the beta value, not the
# over/correct/under-spec framing (that belongs once in the section's prose,
# not every legend entry). Mathtext ($\beta$), not a literal Unicode "β":
# cmr10 (set in set_plot_style) has no Greek glyphs, but its mathtext
# companion cmmi10 does, matching the manuscript's own $\beta$ rendering.
BETA_LABELS = {
    1: r"$\beta=1$",
    2: r"$\beta=2$",
    3: r"$\beta=3$",
}

# Cycled by n-rank (ascending) within whatever n-values a given figure
# actually shows; up to 5 distinct dash patterns before repeating.
LINESTYLES = ["-", "--", ":", "-.", (0, (3, 1, 1, 1))]

_SUFFIX_RE = re.compile(r"^(?P<base>.+) \((?P<suffix>MLE|MAP)\)$")


def _split_suffix(label):
    """Strip a trailing " (MLE)"/" (MAP)" estimator suffix, if present."""
    m = _SUFFIX_RE.match(label)
    return m["base"] if m else label


def linestyles_for(values):
    """{value: linestyle} for sorted unique `values` (typically n), by rank."""
    ordered = sorted(set(values))
    return {v: LINESTYLES[i % len(LINESTYLES)] for i, v in enumerate(ordered)}


def sorted_methods(methods):
    """Order `methods` by METHOD_COLORS's canonical order, unknowns last."""
    order = list(METHOD_COLORS)
    return sorted(set(methods), key=lambda m: order.index(m) if m in order else len(order))


def set_plot_style():
    """Shared seaborn/matplotlib setup for every panel and legend script.

    Explicit point sizes (not a font_scale multiplier), one size (10pt, the
    JCGS minimum) for every text element: figures are embedded at native
    size with no `\\includegraphics[width=...]` downscale, so the authored
    size is the printed size.

    Font family matters as much as point size for visual size: matplotlib's
    default DejaVu Sans has a larger x-height than Computer Modern, the
    family the manuscript's Latin Modern body font is built on, so matching
    point size alone doesn't match apparent size. `cmr10`/`cmmi10` (mathtext)
    are Computer Modern and ship with matplotlib, no new dependency.
    `pdf.fonttype=42` embeds them as CID TrueType, which production
    pipelines are more likely to accept than matplotlib's default Type 3.

    Every rcParam here must be an explicit point size, not a relative
    keyword ("large", "small") -- `fig.supxlabel`/`fig.supylabel` in
    particular use `figure.labelsize`, a separate rcParam from
    `axes.labelsize`/`axes.titlesize` that's easy to miss setting.
    """
    sns.set_style("darkgrid")
    sns.set_style({"legend.frameon": False})
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.labelsize": 10,
            "figure.titlesize": 10,
            "font.family": "serif",
            "font.serif": ["cmr10"],
            "mathtext.fontset": "cm",
            "axes.formatter.use_mathtext": True,
            "pdf.fonttype": 42,
        }
    )


def draw_lines(ax, data, x, y, hue_col, style_col, color_map, marker_map, linestyle_map):
    """Draw one line per (hue_col, style_col) group: median `y` against `x`,
    with an IQR error bar at each point.

    Manual per-group plotting, not a single `sns.lineplot` call, because
    seaborn ties markers to `style`, not `hue`, and this needs marker-by-
    method and dashes-by-n independently (see module docstring).
    """
    hue_order = sorted(
        data[hue_col].unique(), key=lambda h: list(color_map).index(h) if h in color_map else len(color_map)
    )
    for hue_val in hue_order:
        group = data[data[hue_col] == hue_val]
        for style_val in sorted(group[style_col].unique()):
            sub = group[group[style_col] == style_val]
            agg = sub.groupby(x)[y].agg(median="median", q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75))
            agg = agg.sort_index()
            yerr = [agg["median"] - agg["q25"], agg["q75"] - agg["median"]]
            ax.errorbar(
                agg.index,
                agg["median"],
                yerr=yerr,
                color=color_map[hue_val],
                marker=marker_map[hue_val],
                linestyle=linestyle_map[style_val],
                linewidth=1.2,
                markersize=4.5,
                capsize=2,
                elinewidth=0.8,
            )


def method_group_entries(csv_paths, mle_suffix=False):
    """Union the methods and n-values present across one or more CSVs (e.g.
    time_3a + time_3b, sharing one legend) -- returns (methods, ns), each
    sorted, ready for `add_shared_legend`. An MLE/MAP estimator suffix
    on `method`, if present, is stripped first: that's a row facet in
    plot_kl.py, not a hue, so it never belongs in a legend."""
    methods, ns = set(), set()
    for path in csv_paths:
        df = pd.read_csv(path)
        methods.update(df["method"].map(canonical_method_label).map(_split_suffix))
        ns.update(df["n"].astype(int))
    return sorted_methods(methods), sorted(ns)


LEGEND_ROW_H_IN = 0.30
LEGEND_GROUP_GAP_IN = 0.03
LEGEND_PAD_IN = 0.10


def legend_height_in(n_hue_entries, hue_ncol, n_style_entries, style_ncol=None):
    """Height in inches a two-part `add_shared_legend` call will need:
    ceil(n_hue_entries/hue_ncol) rows for the hue grid plus
    ceil(n_style_entries/style_ncol) rows for the style row (usually 1),
    each `LEGEND_ROW_H_IN` tall, plus a small gap and padding.

    Used to size `square_grid`'s `legend_margin` from the actual entry
    counts instead of a guessed fraction.
    """
    n_hue_rows = math.ceil(n_hue_entries / hue_ncol)
    n_style_rows = math.ceil(n_style_entries / (style_ncol or n_style_entries))
    return LEGEND_ROW_H_IN * (n_hue_rows + n_style_rows) + LEGEND_GROUP_GAP_IN + LEGEND_PAD_IN


def square_grid(nrows, ncols, panel_size=1.5, sharex=True, sharey=True):
    """Create an `nrows` x `ncols` grid of square axes (`panel_size` inches
    per side; see `grow_to_fit` for why the actual box size usually ends up
    bigger), sharing the y axis by default (pass `sharey=False` when panels
    plot different metrics) and the x axis by default (pass `sharex=False`
    when panels span genuinely different x-domains), hiding tick labels
    `label_outer` considers redundant given whichever axes are shared.

    Legend space is not reserved here -- call `grow_to_fit` then
    `reserve_legend_margin` after the figure's content is drawn instead.
    """
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(panel_size * ncols, panel_size * nrows),
        sharex=sharex,
        sharey=sharey,
        constrained_layout=True,
        squeeze=False,
    )
    axes = axes.reshape(-1)  # always a flat array, regardless of nrows/ncols
    for ax in axes:
        ax.set_box_aspect(1)
        ax.label_outer()
    return fig, axes


def _grow_to_fit(fig, get_size, set_size, get_box_extent, tol, step_in, max_iter):
    """Grow one figure dimension (`get_size`/`set_size`) until the other,
    `box_aspect`-linked box dimension (`get_box_extent`) stops growing, then
    back off the last (unproductive) step. Shared by `grow_to_fit` and
    `grow_to_fit_width`, which differ only in which dimension they grow.
    """
    fig.canvas.draw()
    prev_extent = get_box_extent()
    prev_size = get_size()
    for _ in range(max_iter):
        set_size(get_size() + step_in)
        fig.canvas.draw()
        extent = get_box_extent()
        if extent - prev_extent < tol:
            set_size(prev_size)  # last step didn't help -- revert the wasted growth
            # A second draw so constrained_layout fully reconverges after the
            # size change -- one draw() alone can leave a label clipped.
            fig.canvas.draw()
            fig.canvas.draw()
            return
        prev_extent, prev_size = extent, get_size()


def grow_to_fit(fig, ref_ax, tol=0.003, step_in=0.3, max_iter=12):
    """Grow `fig`'s height only -- not width -- until `ref_ax`'s box stops
    getting wider.

    `square_grid`'s `ax.set_box_aspect(1)` forces box width to equal box
    height, but the figure height it sets reserves no room for the
    x-tick-labels/`fig.supxlabel` beneath the row, so box height (and thus
    box_aspect-linked width) comes out smaller than `panel_size` even
    though a row's interior columns typically have unused horizontal space
    from `label_outer`. Growing height lets width widen into that same
    space, without touching the page-width-tuned figure width.

    Call after all panel content is drawn (label extents must be final) and
    before `reserve_legend_margin`, so the space measured here is the row's
    own, not inflated by a legend band.
    """
    _grow_to_fit(
        fig, fig.get_figheight, fig.set_figheight, lambda: ref_ax.get_position().width * fig.get_figwidth(),
        tol, step_in, max_iter,
    )


def grow_to_fit_width(fig, ref_ax, tol=0.003, step_in=0.3, max_iter=12):
    """Grow `fig`'s width only -- not height. Mirrors `grow_to_fit`, for a
    single-column figure where the y-label/y-tick text, not the x-axis
    label, is what caps the box below `panel_size`: with only one column,
    there's no `label_outer`-freed neighbor for `grow_to_fit`'s height-only
    growth to unlock horizontal space from. Call before `grow_to_fit` in
    that case, so it can mop up any remaining height-side slack.
    """
    _grow_to_fit(
        fig, fig.get_figwidth, fig.set_figwidth, lambda: ref_ax.get_position().height * fig.get_figheight(),
        tol, step_in, max_iter,
    )


def reserve_legend_margin(fig, loc, frac):
    """Grow `fig` to add space for `add_shared_legend`, and reserve it via
    the constrained-layout engine's `rect` -- without this, a legend added
    afterwards would overlap the grid or get clipped at the figure edge.
    The margin is added on top of `fig`'s current size, not carved out of
    it, so it doesn't shrink the panel area `grow_to_fit` already settled.

    Call after `grow_to_fit`, with `frac` computed relative to `fig`'s
    current (post-growth) size, e.g. `legend_h / (fig.get_figheight() +
    legend_h)` for an "above" legend.
    """
    width, height = fig.get_figwidth(), fig.get_figheight()
    if loc == "above":
        height /= 1 - frac
    elif loc == "beside":
        width /= 1 - frac
    else:
        raise ValueError(f"Unknown legend loc: {loc!r}")
    fig.set_size_inches(width, height)
    rect = {"above": (0.0, 0.0, 1.0, 1.0 - frac), "beside": (0.0, 0.0, 1.0 - frac, 1.0)}[loc]
    fig.get_layout_engine().set(rect=rect)
    # Two draws so constrained_layout fully reconverges after the size
    # change (see grow_to_fit's revert branch).
    fig.canvas.draw()
    fig.canvas.draw()


def _fit_legend(fig, handles, labels, max_ncol, max_width_in, x, y, anchor_loc="upper left", **kwargs):
    """Place a legend at up to `max_ncol` columns, anchored at figure-
    fraction `(x, y)` via `anchor_loc` (e.g. "upper left" to grow rightward
    from `x`, "upper center" to grow symmetrically around `x`), backing off
    to fewer columns until its actual rendered width fits `max_width_in`,
    or there's only one column left to try.

    A legend's rendered width is set by its content, not clipped to the
    parent figure's nominal size, so it's measured via a canvas draw rather
    than trusted against the requested `max_width_in`. Since
    `add_shared_legend` doesn't save with `bbox_inches="tight"`, anything
    past the canvas edge is silently clipped at save time rather than
    visibly overflowing -- catching that here, at generation time, matters.
    """
    ncol = max(1, min(max_ncol, len(handles)))
    while True:
        legend = fig.legend(handles, labels, loc=anchor_loc, bbox_to_anchor=(x, y), ncol=ncol, **kwargs)
        fig.canvas.draw()
        bbox = legend.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
        width_in = bbox.width * fig.get_figwidth()
        if width_in <= max_width_in:
            return legend, bbox
        if ncol == 1:
            # Can't back off any further -- some single label is wider than
            # max_width_in on its own.
            raise ValueError(
                f"_fit_legend: content needs {width_in:.2f}in but only {max_width_in:.2f}in is reserved, "
                f"even at ncol=1 -- widen the legend's frac/max_width_in, or shorten its longest label."
            )
        legend.remove()
        ncol -= 1


def add_shared_legend(fig, hue_entries, style_entries, loc, frac, hue_ncol=1, fontsize=None):
    """Draw a two-part legend (method/beta-arm grid + n/linestyle row)
    directly into `fig`'s margin reserved by `square_grid`'s
    `legend_margin=(loc, frac)` -- same `loc`/`frac` values as passed there.

    Two physically separate `fig.legend` calls, not one legend listing every
    method x n combination, since marker/linestyle carry method/n
    independently (see module docstring): each dimension only needs listing
    once, letting a multi-panel figure share a single legend instead of
    repeating it per panel.

    `hue_entries`: [(label, color, marker), ...].
    `style_entries`: [(label, linestyle), ...].
    `hue_ncol`: requested columns for the method/beta-arm grid; `_fit_legend`
    may reduce it if that doesn't fit the reserved margin.
    """
    fig_w, fig_h = fig.get_figwidth(), fig.get_figheight()
    if loc == "above":
        x0, y0, x1, y1 = 0.0, 1.0 - frac, 1.0, 1.0
        # Centered, not left-anchored, so a narrower row doesn't look
        # flush-left by accident; each row centers independently.
        anchor_loc, legend_x = "upper center", (x0 + x1) / 2
    elif loc == "beside":
        x0, y0, x1, y1 = 1.0 - frac, 0.0, 1.0, 1.0
        anchor_loc, legend_x = "upper left", x0
    else:
        raise ValueError(f"Unknown legend loc: {loc!r}")
    max_width_in = (x1 - x0) * fig_w

    hue_handles = [
        plt.Line2D([0], [0], color=color, marker=marker, linestyle="-", markersize=5, linewidth=1.2)
        for _, color, marker in hue_entries
    ]
    hue_labels = [label for label, _, _ in hue_entries]

    style_handles = [
        plt.Line2D([0], [0], color="0.3", marker="None", linestyle=linestyle, linewidth=1.4)
        for _, linestyle in style_entries
    ]
    style_labels = [label for label, _ in style_entries]

    # handlelength=1.484em is chosen, not round: at this legend's font/
    # linewidth it's an exact multiple of every non-solid linestyle's dash
    # period, so each line sample ends cleanly instead of on a stray stub.
    kwargs = dict(
        frameon=False, fontsize=fontsize, handlelength=1.484, handletextpad=0.4, columnspacing=0.8, labelspacing=0.4
    )
    vgap_in = LEGEND_GROUP_GAP_IN  # keep in sync with legend_height_in's margin-sizing estimate

    hue_legend, hue_bbox = _fit_legend(
        fig, hue_handles, hue_labels, hue_ncol, max_width_in, x=legend_x, y=y1, anchor_loc=anchor_loc, **kwargs
    )
    _, style_bbox = _fit_legend(
        fig,
        style_handles,
        style_labels,
        len(style_entries),
        max_width_in,
        x=legend_x,
        y=hue_bbox.y0 - vgap_in / fig_h,
        anchor_loc=anchor_loc,
        **kwargs,
    )

    # Catch a margin sized too small for its content at generation time
    # rather than needing a human to spot the overlap in the rendered PDF.
    if style_bbox.y0 < y0 - 1e-6:
        overflow_in = (y0 - style_bbox.y0) * fig_h
        raise ValueError(
            f"add_shared_legend: legend needs {overflow_in:.2f}in more than its reserved margin "
            f"(loc={loc!r}, frac={frac}) -- it will overlap the axes. Widen `frac` (see "
            f"`legend_height_in` for a content-derived value) or reduce hue_ncol/entry count."
        )

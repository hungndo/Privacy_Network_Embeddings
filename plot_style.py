"""
Shared plotting style for every analysis notebook in this repo.

Why this file exists
--------------------
Each notebook used to build its palette with `plt.cm.tab10.colors` indexed by
the position of a method in `sorted(df["method"].unique())`.  That makes the
colour of a method depend on WHICH OTHER METHODS happen to be in that CSV:
sbm_dpsgd came out orange in the polblogs figures (2 methods) but green in the
known-label ones (3 methods), and edge-flip's blue was the only stable colour.
Here a method's colour is keyed off the method NAME, so it is identical in
every figure of the thesis.

Three rules this module enforces:

  1. One colour per method, everywhere (METHOD_COLORS).
  2. One display name per method (METHOD_LABELS) -- in particular
     `sbm_dpsgd_all_noised` is shown as "SBM-DPSGD"; the "all noised"
     implementation detail never reaches a figure.
  3. Box medians are drawn in MEDIAN_COLOR, which is deliberately NOT any
     method colour. Matplotlib's default median is orange, the same orange
     several boxes were filled with, so medians read as part of the fill.

Usage in a notebook cell:

    from plot_style import (apply_style, grouped_boxplot, method_legend,
                            method_color, method_label, MEDIAN_COLOR)
    apply_style()
    fig, ax = plt.subplots(figsize=figsize(8, 5))
    grouped_boxplot(ax, sub, "nmi", methods=methods)
    method_legend(ax, methods, extra_true_value="perfect (NMI=1)")
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


# ══════════════════════════════════════════════════════════════════════════
# Palette
# ══════════════════════════════════════════════════════════════════════════
# Method families share a hue: everything edge-flip is blue, everything
# SBM-DPSGD is purple. Red is reserved for medians and is never a method.

C_EDGEFLIP        = "#4C72B0"   # blue
C_EDGEFLIP_ALT    = "#D4A017"   # goldenrod -- the second edge-flip variant.
                                # Deliberately NOT another blue: the pure-DP
                                # and approximate-DP arms are drawn side by
                                # side in the appendix figure, so they need
                                # maximum separation, not a shared hue.
C_GAUSSIAN        = "#55A868"   # green
C_SBM_DPSGD       = "#C2549D"   # magenta. NOT purple: purple sat too close to
                                # the edge-flip blue to separate at a glance,
                                # and these two are the head-to-head pair in
                                # most figures.
C_SPECTRAL        = "#937860"   # brown -- non-private baseline

#: Median line colour. Not a method colour, and not orange (the matplotlib
#: default, which collided with the box fills). Near-black rather than a hue:
#: it has to stay legible on top of EVERY box fill, and any coloured choice
#: eventually lands next to a fill it cannot be seen against (firebrick, the
#: previous pick, disappeared into the magenta boxes).
MEDIAN_COLOR = "#1A1A1A"

#: Reference lines (true value, NMI = 1, zero line). Mid-grey and dashed, so
#: it never competes with the near-black solid medians.
REFERENCE_COLOR = "#4D4D4D"

METHOD_COLORS = {
    # ── edge-flip family ──────────────────────────────────────────────────
    "edge_flip":            C_EDGEFLIP,
    "edge_flip_vem":        C_EDGEFLIP,
    "edge_flip_pure_dp":    C_EDGEFLIP,
    "edge_flip_approx_dp":  C_EDGEFLIP_ALT,
    # ── Gaussian mechanism ────────────────────────────────────────────────
    "gaussian":             C_GAUSSIAN,
    # ── our method ────────────────────────────────────────────────────────
    "sbm_dpsgd":            C_SBM_DPSGD,
    "sbm_dpsgd_all_noised": C_SBM_DPSGD,
    # ── non-private baseline ──────────────────────────────────────────────
    "spectral":             C_SPECTRAL,
}

METHOD_LABELS = {
    "edge_flip":            "Edge-flip",
    "edge_flip_vem":        "Edge-flip",
    "edge_flip_pure_dp":    "Edge-flip (Pure DP)",
    "edge_flip_approx_dp":  "Edge-flip (Approx. DP)",
    "gaussian":             "Gaussian",
    "sbm_dpsgd":            "SBM-DPSGD",
    "sbm_dpsgd_all_noised": "SBM-DPSGD",     # never print "all_noised"
    "spectral":             "Spectral (non-private)",
}

#: Stable drawing order: baselines first, our method last, non-private last of
#: all. Methods not listed keep their input order after these.
METHOD_ORDER_HINT = [
    "edge_flip", "edge_flip_vem", "edge_flip_pure_dp", "edge_flip_approx_dp",
    "gaussian", "sbm_dpsgd", "sbm_dpsgd_all_noised", "spectral",
]

_FALLBACK_COLORS = ["#C8A2C8", "#BFBFBF", "#CCB974", "#DA8BC3"]


def method_color(method):
    """Colour for a method name, stable across every figure."""
    if method in METHOD_COLORS:
        return METHOD_COLORS[method]
    idx = abs(hash(method)) % len(_FALLBACK_COLORS)
    return _FALLBACK_COLORS[idx]


def method_label(method):
    """Display name for a method name (falls back to a tidied raw name)."""
    return METHOD_LABELS.get(method, str(method).replace("_", " "))


def order_methods(methods):
    """Sort methods into the canonical drawing order."""
    methods = list(methods)
    known = [m for m in METHOD_ORDER_HINT if m in methods]
    rest = sorted(m for m in methods if m not in METHOD_ORDER_HINT)
    return known + rest


# ══════════════════════════════════════════════════════════════════════════
# Sizing
# ══════════════════════════════════════════════════════════════════════════
#: Every figsize in the notebooks is multiplied by this, so figures (and
#: therefore the text on them, which is set in points) come out larger.
FIG_SCALE = 1.35

BOX_ALPHA = 0.65


def figsize(w, h):
    """Scale a (width, height) pair by FIG_SCALE."""
    return (w * FIG_SCALE, h * FIG_SCALE)


def apply_style():
    """Set rcParams: larger type everywhere, especially axis labels/ticks."""
    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",

        "font.size": 15,
        "axes.titlesize": 17,
        "axes.labelsize": 17,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 14,
        "legend.title_fontsize": 14,
        "figure.titlesize": 20,

        "axes.titlepad": 10,
        "axes.labelpad": 6,
        "axes.linewidth": 1.1,
        "xtick.major.width": 1.1,
        "ytick.major.width": 1.1,
        "xtick.major.size": 5,
        "ytick.major.size": 5,

        "axes.grid": False,
        "grid.alpha": 0.3,
        "legend.framealpha": 0.9,
        "figure.autolayout": False,
    })


# ══════════════════════════════════════════════════════════════════════════
# The grouped boxplot every notebook draws
# ══════════════════════════════════════════════════════════════════════════

def grouped_boxplot(ax, data, value_col, x_col="epsilon", method_col="method",
                    methods=None, x_values=None, box_frac=0.8,
                    x_fmt="{:.2g}", rotation=45, showfliers=True):
    """
    One box per (x value, method) cell, methods side by side within each x
    group -- the layout shared by every figure in this repo.

    Colours come from the METHOD NAME (not its index), medians are drawn in
    MEDIAN_COLOR, and x tick labels are written from the x values.

    Returns the list of x values actually plotted.
    """
    methods = order_methods(methods if methods is not None
                            else data[method_col].unique())
    if x_values is None:
        x_values = sorted(data[x_col].unique())
    width = box_frac / max(len(methods), 1)

    boxes, positions, colors = [], [], []
    for x_idx, xv in enumerate(x_values):
        for m_idx, method in enumerate(methods):
            vals = data.loc[(data[x_col] == xv) & (data[method_col] == method),
                            value_col].dropna()
            if len(vals) == 0:
                continue
            boxes.append(vals.values)
            positions.append(x_idx + (m_idx - len(methods) / 2 + 0.5) * width)
            colors.append(method_color(method))

    if boxes:
        bp = ax.boxplot(
            boxes, positions=positions, widths=width * 0.85, patch_artist=True,
            medianprops=dict(color=MEDIAN_COLOR, linewidth=2.0),
            whiskerprops=dict(linewidth=1.1),
            capprops=dict(linewidth=1.1),
            flierprops=dict(marker="o", markersize=3.5, alpha=0.55,
                            markerfacecolor="none", markeredgewidth=0.8),
            showfliers=showfliers,
        )
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(BOX_ALPHA)
            patch.set_edgecolor("#333333")
            patch.set_linewidth(0.9)

    ax.set_xticks(range(len(x_values)))
    ax.set_xticklabels([x_fmt.format(x) for x in x_values], rotation=rotation)
    ax.set_xlim(-0.6, len(x_values) - 0.4)
    ax.grid(alpha=0.3)
    return x_values


def method_legend(ax, methods, reference_label=None, show_median=True,
                  extra_handles=None, **kwargs):
    """
    Legend with one patch per method (canonical colour + display name), plus
    optional entries for the reference line and the median line.
    """
    methods = order_methods(methods)
    handles = [Patch(facecolor=method_color(m), alpha=BOX_ALPHA,
                     edgecolor="#333333", label=method_label(m))
               for m in methods]
    if show_median:
        handles.append(Line2D([0], [0], color=MEDIAN_COLOR, linewidth=2.0,
                              label="median"))
    if reference_label:
        handles.append(Line2D([0], [0], color=REFERENCE_COLOR, linestyle="--",
                              linewidth=1.5, label=reference_label))
    if extra_handles:
        handles.extend(extra_handles)
    kwargs.setdefault("fontsize", 13)
    return ax.legend(handles=handles, **kwargs)


def reference_line(ax, y, label=None, axis="h"):
    """Dashed black reference line (true value / NMI = 1 / zero)."""
    fn = ax.axhline if axis == "h" else ax.axvline
    return fn(y, color=REFERENCE_COLOR, linestyle="--", alpha=0.75,
              linewidth=1.6, label=label)

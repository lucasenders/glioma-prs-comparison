"""Names, order, colours, data loading and saving shared by Figures 2 to 7, read from pipeline/summary/out/."""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Folder paths
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(os.path.dirname(HERE)), "pipeline", "summary", "out")

# Figure width
CM = 1 / 2.54
WIDTH = 17 * CM

# Display name, colour and marker of each method, in thesis order
METHODS = [
    ("melin_25", "Melin", "#7F7F7F", "o"),
    ("eckel_passow_25", "Eckel-Passow", "#882255", "o"),
    ("lassosum", "lassosum", "#661100", "o"),
    ("ldpred2", "LDpred2", "#117733", "o"),
    ("prs_cs", "PRS-CS", "#999933", "o"),
    ("standard_prs", "Standard PRS", "#000000", "o"),
    ("elastic_net", "Elastic net", "#0072B2", "s"),
    ("random_forest", "Random forest", "#D55E00", "^"),
    ("xgboost_model", "XGBoost", "#E69F00", "D"),
    ("svm", "SVM", "#CC79A7", "v"),
    ("mlp", "MLP 256-64-16", "#009E73", "P"),
    ("mlp_narrow", "MLP 64-16-4", "#332288", "o"),
    ("mlp_shallow", "MLP 256-64", "#44AA99", "o"),
    ("mlp_small", "MLP 64-16", "#88CCEE", "o"),
]
NAME = {k: n for k, n, _, _ in METHODS}
COLOUR = {k: c for k, _, c, _ in METHODS}
MARKER = {k: m for k, _, _, m in METHODS}

# Feature sets: the two fixed sets, then the selected sets by mean number of variants
SETS = [
    ("25 variant set", "25-variant set", "25-variant\nset"),
    ("50 variant set", "50-variant set", "50-variant\nset"),
    ("ct_1e-05", "C+T 1×10−5", "C+T\n$1\\times10^{-5}$"),
    ("threshold_1e-05", "Threshold 1×10−5", "Threshold\n$1\\times10^{-5}$"),
    ("ct_0.0001", "C+T 1×10−4", "C+T\n$1\\times10^{-4}$"),
    ("threshold_0.0001", "Threshold 1×10−4", "Threshold\n$1\\times10^{-4}$"),
    ("ct_0.001", "C+T 1×10−3", "C+T\n$1\\times10^{-3}$"),
    ("threshold_0.001", "Threshold 1×10−3", "Threshold\n$1\\times10^{-3}$"),
]
SET_KEYS = [k for k, _, _ in SETS]
SET_NAME = {k: n for k, n, _ in SETS}
SET_TICK = {k: t for k, _, t in SETS}

# Datasets, with panel letter and title
DATASETS = [("AllGlioma", "A", "All glioma"), ("GBM", "B", "GBM"), ("nonGBM", "C", "Non-GBM")]

# Axis colours
SPINE, GRID, TICK_TEXT = "#b0b0b0", "#e3e3e3", "#333333"


def setup():
    """Set the fonts and line widths: Arial, 8 pt, with editable text in the SVG."""

    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans"],
        "font.size": 8, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial", "mathtext.default": "regular",
        "svg.fonttype": "none", "pdf.fonttype": 42, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.color": "#555555", "ytick.color": "#555555", "xtick.labelcolor": TICK_TEXT, "ytick.labelcolor": TICK_TEXT,
        "legend.frameon": False,
    })


def summary(metric):
    """Read the rows of auc_summary.csv for one metric."""

    s = pd.read_csv(os.path.join(OUT, "auc_summary.csv"))
    return s[s["metric"] == metric].reset_index(drop=True)


def compare(name, metric="test_auc"):
    """Read the rows of compare_vs_<name>.csv for one metric."""

    c = pd.read_csv(os.path.join(OUT, f"compare_vs_{name}.csv"))
    return c[c["metric"] == metric].reset_index(drop=True)


def label_panel(ax, letter, title):
    """Write the dataset name above a panel."""

    ax.set_title(title, loc="center", fontsize=9, fontweight="bold", pad=4)


def tidy(ax, left=True, bottom=True):
    """Hide the top and right axis lines and grey the others."""

    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(SPINE)


def save(fig, stem):
    """Write <stem>.svg and <stem>.png at 300 dpi beside the scripts."""

    fig.savefig(os.path.join(HERE, f"{stem}.svg"))
    fig.savefig(os.path.join(HERE, f"{stem}.png"), dpi=300)
    print(f"wrote {stem}.svg and {stem}.png")


def write_values(stem, frame):
    """Write the plotted values as a TSV beside the figure."""

    frame.to_csv(os.path.join(HERE, f"{stem}_values.tsv"), sep="\t", index=False)


# Groups of feature sets in Figures 2 and 6, and the position of the divider between them
GROUPS = [slice(0, 2), slice(2, len(SETS))]
DIVIDER_X = 1.5


def divider(ax):
    """Draw a dashed vertical line between the fixed sets and the selected sets."""

    ax.axvline(DIVIDER_X, color="#9a9a9a", lw=0.6, ls=(0, (3, 2)), zorder=1)


def plot_grouped(ax, x, y, **style):
    """Plot a line in each group separately, with no segment crossing the divider."""

    for g in GROUPS:
        ax.plot(x[g], y[g], **style)

"""Figure 6: mean test AUC and mean training AUC of the standard PRS and the ML and DL models across feature sets. Output goes to Figure6.svg and Figure6.png beside this script.

Run with: python3 figure6.py
"""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import common as c

# Methods drawn, and the y-axis range and ticks
LINES = ["standard_prs", "elastic_net", "random_forest", "xgboost_model", "svm", "mlp"]
Y_LOW, Y_HIGH = 0.45, 1.0
Y_TICKS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# Read the mean test and training AUC and set up one panel per dataset
c.setup()
test = c.summary("test_auc").set_index(["model", "feature_set", "dataset"])["mean"]
train = c.summary("train_auc").set_index(["model", "feature_set", "dataset"])["mean"]
fig, axes = plt.subplots(3, 1, figsize=(c.WIDTH, 20 * c.CM), sharex=True)
x = list(range(len(c.SET_KEYS)))
rows = []
for ax, (dataset, letter, title) in zip(axes, c.DATASETS):
    # Draw the grid lines and the divider between the fixed and selected sets
    for level in Y_TICKS:
        ax.axhline(level, color=c.GRID, lw=0.5, zorder=1)
    c.divider(ax)

    # Draw each method's training AUC dashed and test AUC solid
    for model in LINES:
        yt = [test[(model, s, dataset)] for s in c.SET_KEYS]
        yr = [train[(model, s, dataset)] for s in c.SET_KEYS]
        assert min(yt + yr) >= Y_LOW and max(yt + yr) <= Y_HIGH, (model, dataset)
        c.plot_grouped(ax, x, yr, color=c.COLOUR[model], lw=1.1, ls="--", marker=c.MARKER[model], ms=3, mew=0, zorder=3)
        c.plot_grouped(ax, x, yt, color=c.COLOUR[model], lw=1.2, marker=c.MARKER[model], ms=3.2, mew=0, zorder=4)
        rows += [{"Dataset": title, "Method": c.NAME[model], "Feature set": c.SET_NAME[s], "Mean test AUC": a, "Mean training AUC": b}
                 for s, a, b in zip(c.SET_KEYS, yt, yr)]

    # Label the panel and its axes
    c.label_panel(ax, letter, title)
    ax.set_xlim(-0.5, len(x) - 0.5)
    ax.set_ylim(Y_LOW, Y_HIGH + 0.02)
    ax.set_yticks(Y_TICKS)
    ax.set_yticklabels([f"{v:.1f}" for v in Y_TICKS])
    ax.set_xticks(x)
    ax.set_xticklabels([c.SET_TICK[s] for s in c.SET_KEYS])
    ax.tick_params(axis="x", pad=3)
    c.tidy(ax)
    ax.set_ylabel("Mean AUC")
axes[-1].set_xlabel("Feature set")

# Add the legend and save the figure and its values
handles = [Line2D([], [], color=c.COLOUR[m], lw=1.2, marker=c.MARKER[m], ms=3.2, mew=0, label=c.NAME[m]) for m in LINES]
handles += [Line2D([], [], color="#333333", lw=1.2, label="Test AUC (solid)"),
            Line2D([], [], color="#333333", lw=1.1, ls="--", label="Training AUC (dashed)")]
fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.53, 1.0), handlelength=2.2, columnspacing=1.6)
fig.subplots_adjust(left=0.085, right=0.99, top=0.915, bottom=0.07, hspace=0.3)
c.save(fig, "Figure6")
c.write_values("Figure6", pd.DataFrame(rows))

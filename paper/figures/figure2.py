"""Figure 2: mean test AUC of the standard PRS and the ML and DL models across feature sets. Output goes to Figure2.svg and Figure2.png beside this script.

Run with: python3 figure2.py
"""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import common as c

# Methods drawn, and the y-axis range and ticks
LINES = ["standard_prs", "elastic_net", "random_forest", "xgboost_model", "svm", "mlp"]
Y_LOW, Y_HIGH = 0.45, 0.70
Y_TICKS = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

# Read the mean test AUC and set up one panel per dataset
c.setup()
test = c.summary("test_auc").set_index(["model", "feature_set", "dataset"])["mean"]
fig, axes = plt.subplots(3, 1, figsize=(c.WIDTH, 19 * c.CM), sharex=True)
x = list(range(len(c.SET_KEYS)))
rows = []
for ax, (dataset, letter, title) in zip(axes, c.DATASETS):
    # Draw the grid lines and the divider between the fixed and selected sets
    for level in Y_TICKS:
        ax.axhline(level, color=c.GRID, lw=0.5, zorder=1)
    c.divider(ax)

    # Draw the Melin score as a dashed line
    melin = test[("published", "melin_25", dataset)]
    ax.axhline(melin, color=c.COLOUR["melin_25"], lw=0.9, ls="--", zorder=2)
    ax.text(x[-1] + 0.45, melin + 0.004, f"Melin score {melin:.3f}", ha="right", va="bottom", fontsize=8, color=c.COLOUR["melin_25"])

    # Draw each method across the feature sets
    for model in LINES:
        y = [test[(model, s, dataset)] for s in c.SET_KEYS]
        assert min(y) >= Y_LOW and max(y) <= Y_HIGH, (model, dataset, min(y), max(y))
        c.plot_grouped(ax, x, y, color=c.COLOUR[model], lw=1.2, marker=c.MARKER[model], ms=3.2, mew=0, zorder=3)
        rows += [{"Dataset": title, "Method": c.NAME[model], "Feature set": c.SET_NAME[s], "Mean test AUC": v} for s, v in zip(c.SET_KEYS, y)]

    # Label the panel and its axes
    c.label_panel(ax, letter, title)
    ax.set_xlim(-0.5, len(x) - 0.5)
    ax.set_ylim(Y_LOW, Y_HIGH)
    ax.set_yticks(Y_TICKS)
    ax.set_yticklabels([f"{v:.2f}" for v in Y_TICKS])
    ax.set_xticks(x)
    ax.set_xticklabels([c.SET_TICK[s] for s in c.SET_KEYS])
    ax.tick_params(axis="x", pad=3)
    c.tidy(ax)
    ax.set_ylabel("Mean test AUC")
axes[-1].set_xlabel("Feature set")

# Add the legend and save the figure and its values
handles = [Line2D([], [], color=c.COLOUR[m], lw=1.2, marker=c.MARKER[m], ms=3.2, mew=0, label=c.NAME[m]) for m in LINES]
handles.append(Line2D([], [], color=c.COLOUR["melin_25"], lw=0.9, ls="--", label="Melin score"))
fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.53, 1.0), handlelength=2.2, columnspacing=1.6)
fig.subplots_adjust(left=0.085, right=0.99, top=0.915, bottom=0.075, hspace=0.3)
c.save(fig, "Figure2")
c.write_values("Figure2", pd.DataFrame(rows))

"""Figure 7: mean test AUC of each method in all test-fold participants and in those aged 40 to 64, with 95% CIs. Output goes to Figure7.svg and Figure7.png beside this script.

Run with: python3 figure7.py
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import pandas as pd
import common as c

# Rows drawn: display key, model and feature set in the summary file
ROWS = [("melin_25", "published", "melin_25"), ("eckel_passow_25", "published", "eckel_passow_25"),
        ("lassosum", "lassosum", "genome_wide"), ("ldpred2", "ldpred2", "genome_wide"), ("prs_cs", "prs_cs", "genome_wide"),
        ("standard_prs", "standard_prs", "25 variant set"), ("elastic_net", "elastic_net", "25 variant set"),
        ("random_forest", "random_forest", "25 variant set"), ("xgboost_model", "xgboost_model", "25 variant set"),
        ("svm", "svm", "25 variant set"), ("mlp", "mlp", "25 variant set"), ("mlp_narrow", "mlp_narrow", "25 variant set"),
        ("mlp_shallow", "mlp_shallow", "25 variant set"), ("mlp_small", "mlp_small", "25 variant set")]

# X-axis ticks and bar height
X_TICKS = [0.0, 0.2, 0.4, 0.6, 0.8]
H = 0.36

# Read the test AUC in everyone and in the age band, and set up one panel per dataset
c.setup()
plt.rcParams["hatch.linewidth"] = 0.5
everyone = c.summary("test_auc").set_index(["model", "feature_set", "dataset"])
band = c.summary("age_40_64_auc").set_index(["model", "feature_set", "dataset"])
fig, axes = plt.subplots(1, 3, figsize=(c.WIDTH, 16 * c.CM), sharey=True)
ypos = list(range(len(ROWS)))[::-1]
rows = []
for ax, (dataset, letter, title) in zip(axes, c.DATASETS):
    # Draw the grid lines and the line at an AUC of 0.5
    for level in X_TICKS:
        ax.axvline(level, color=c.GRID, lw=0.5, zorder=1)
    ax.axvline(0.5, color="#333333", lw=0.8, ls="--", zorder=2)

    # Draw each method's two bars and CIs, solid for everyone and hatched for the age band
    for y, (key, model, fset) in zip(ypos, ROWS):
        if (model, fset, dataset) not in everyone.index:
            continue
        a, b = everyone.loc[(model, fset, dataset)], band.loc[(model, fset, dataset)]
        colour = c.COLOUR[key]
        ax.barh(y + H / 2, a["mean"], height=H, color=colour, lw=0, zorder=3)
        ax.barh(y - H / 2, b["mean"], height=H, facecolor="white", edgecolor=colour, hatch="/////", lw=0.5, zorder=3)
        ax.plot([a["lo"], a["hi"]], [y + H / 2] * 2, color="black", lw=0.7, zorder=4)
        ax.plot([b["lo"], b["hi"]], [y - H / 2] * 2, color="black", lw=0.7, zorder=4)
        rows.append({"Dataset": title, "Method": c.NAME[key], "All: mean": a["mean"], "All: lower": a["lo"], "All: upper": a["hi"],
                     "Aged 40 to 64: mean": b["mean"], "Aged 40 to 64: lower": b["lo"], "Aged 40 to 64: upper": b["hi"]})

    # Label the panel and its axes
    c.label_panel(ax, letter, title)
    ax.set_xlim(0, 0.9)
    ax.set_xticks(X_TICKS)
    ax.set_xticklabels([f"{v:.1f}".replace("-", "−") for v in X_TICKS])
    ax.set_ylim(-0.6, len(ROWS) - 0.4)
    ax.set_yticks(ypos)
    ax.set_yticklabels([c.NAME[k] for k, _, _ in ROWS])
    ax.tick_params(axis="y", length=0)
    c.tidy(ax, left=False)
axes[1].set_xlabel("Mean test AUC")

# Add the legend and save the figure and its values
handles = [Patch(facecolor="#555555", lw=0, label="All test-fold participants"),
           Patch(facecolor="white", edgecolor="#555555", hatch="/////", lw=0.5, label="Aged 40 to 64"),
           Line2D([], [], color="black", lw=0.7, label="95% CI"),
           Line2D([], [], color="#333333", lw=0.8, ls="--", label="AUC of 0.5")]
fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.56, 0.0), columnspacing=2)
fig.subplots_adjust(left=0.15, right=0.975, top=0.955, bottom=0.12, wspace=0.12)
c.save(fig, "Figure7")
c.write_values("Figure7", pd.DataFrame(rows))

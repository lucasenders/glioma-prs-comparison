"""Figure 3: paired difference in test AUC from the Melin score for each method, with 95% CIs. Output goes to Figure3.svg and Figure3.png beside this script.

Run with: python3 figure3.py
"""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import common as c

# Rows drawn: display key, model and feature set in the comparison file
ROWS = [("eckel_passow_25", "published", "eckel_passow_25"),
        ("lassosum", "lassosum", "genome_wide"), ("ldpred2", "ldpred2", "genome_wide"), ("prs_cs", "prs_cs", "genome_wide"),
        ("standard_prs", "standard_prs", "25 variant set"), ("elastic_net", "elastic_net", "25 variant set"),
        ("random_forest", "random_forest", "25 variant set"), ("xgboost_model", "xgboost_model", "25 variant set"),
        ("svm", "svm", "25 variant set"), ("mlp", "mlp", "25 variant set"), ("mlp_narrow", "mlp_narrow", "25 variant set"),
        ("mlp_shallow", "mlp_shallow", "25 variant set"), ("mlp_small", "mlp_small", "25 variant set")]

# X-axis range and ticks
X_LOW, X_HIGH = -0.30, 0.10
X_TICKS = [-0.3, -0.2, -0.1, 0.0, 0.1]

# Read the differences from Melin and set up one panel per dataset
c.setup()
diff = c.compare("published").set_index(["model", "feature_set", "dataset"])
fig, axes = plt.subplots(1, 3, figsize=(c.WIDTH, 10 * c.CM), sharey=True)
ypos = list(range(len(ROWS)))[::-1]
rows = []
for ax, (dataset, letter, title) in zip(axes, c.DATASETS):
    # Draw the zero line
    ax.axvline(0, color="#333333", lw=0.8, ls="--", zorder=2)

    # Draw each method's difference and CI, filled where the CI excludes zero
    for y, (key, model, fset) in zip(ypos, ROWS):
        if (model, fset, dataset) not in diff.index:
            continue
        r = diff.loc[(model, fset, dataset)]
        excludes = r["hi"] < 0 or r["lo"] > 0
        colour = c.COLOUR[key]
        ax.plot([r["lo"], r["hi"]], [y, y], color=colour, lw=1.1, zorder=3, solid_capstyle="butt")
        ax.plot(r["difference"], y, marker="o", ms=4.5, mfc=colour if excludes else "white", mec=colour, mew=1.1, zorder=4)
        rows.append({"Dataset": title, "Method": c.NAME[key], "Difference": r["difference"], "Lower": r["lo"], "Upper": r["hi"], "CI excludes zero": excludes})

    # Label the panel and its axes, and draw the grid lines
    c.label_panel(ax, letter, title)
    ax.set_xlim(X_LOW, X_HIGH)
    ax.set_xticks(X_TICKS)
    ax.set_xticklabels([f"{v:.1f}".replace("-", "−") for v in X_TICKS])
    ax.set_ylim(-0.7, len(ROWS) - 0.3)
    ax.set_yticks(ypos)
    ax.set_yticklabels([c.NAME[k] for k, _, _ in ROWS])
    ax.tick_params(axis="y", length=0)
    c.tidy(ax, left=False)
    for level in X_TICKS:
        ax.axvline(level, color=c.GRID, lw=0.5, zorder=1)
axes[1].set_xlabel("Difference in test AUC from the Melin score")

# Add the legend and save the figure and its values
handles = [Line2D([], [], color="#333333", lw=1.1, marker="o", ms=4.5, mfc="#333333", mec="#333333", label="95% CI excludes zero"),
           Line2D([], [], color="#333333", lw=1.1, marker="o", ms=4.5, mfc="white", mec="#333333", label="95% CI includes zero")]
fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.56, 0.0), columnspacing=3)
fig.subplots_adjust(left=0.15, right=0.975, top=0.93, bottom=0.21, wspace=0.25)
c.save(fig, "Figure3")
c.write_values("Figure3", pd.DataFrame(rows))

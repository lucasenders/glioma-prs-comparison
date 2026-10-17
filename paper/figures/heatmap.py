"""Heat map drawing shared by Figures 4 and 5: paired differences per model and feature set, one panel per dataset."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import common as c


def draw(models, comparison, colourbar_label, stem, height):
    """Draw three heat maps of model minus reference, with a dot where the 95% CI excludes zero."""

    # Read the differences and set the colour range
    c.setup()
    diff = c.compare(comparison).set_index(["model", "feature_set", "dataset"])
    limit = np.ceil(diff["difference"].abs().max() / 0.02) * 0.02
    fig, axes = plt.subplots(1, 3, figsize=(c.WIDTH, height * c.CM), sharey=True)
    rows = []
    for ax, (dataset, letter, title) in zip(axes, c.DATASETS):
        # Draw the heat map
        grid = np.array([[diff.loc[(m, s, dataset), "difference"] for s in c.SET_KEYS] for m in models])
        image = ax.imshow(grid, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")

        # Mark each cell whose CI excludes zero
        for i, m in enumerate(models):
            for j, s in enumerate(c.SET_KEYS):
                r = diff.loc[(m, s, dataset)]
                excludes = r["hi"] < 0 or r["lo"] > 0
                if excludes:
                    ax.plot(j, i, marker="o", ms=3, color="black", mew=0)
                rows.append({"Dataset": title, "Method": c.NAME[m], "Feature set": c.SET_NAME[s], "Difference": r["difference"],
                             "Lower": r["lo"], "Upper": r["hi"], "CI excludes zero": excludes})

        # Label the panel and its axes, with white lines between cells
        ax.set_xticks(range(len(c.SET_KEYS)), minor=False)
        ax.set_xticks(np.arange(-0.5, len(c.SET_KEYS)), minor=True)
        ax.set_yticks(range(len(models)), minor=False)
        ax.set_yticks(np.arange(-0.5, len(models)), minor=True)
        ax.grid(which="minor", color="white", lw=1.5)
        ax.tick_params(which="both", length=0)
        ax.set_xticklabels([c.SET_TICK[s].replace("\n", " ") for s in c.SET_KEYS], rotation=90, ha="center", va="top")
        ax.set_yticklabels([c.NAME[m] for m in models])
        for side in ax.spines.values():
            side.set_visible(False)
        c.label_panel(ax, letter, title)

    # Add the colour bar and the legend
    bar = fig.add_axes([0.30, 0.09, 0.36, 0.025])
    cb = fig.colorbar(image, cax=bar, orientation="horizontal")
    cb.outline.set_visible(False)
    cb.set_label(colourbar_label, fontsize=9)
    cb.ax.tick_params(length=2.5, labelsize=8)
    handle = Line2D([], [], marker="o", ms=3, color="black", lw=0, label="95% CI excludes zero")
    fig.legend(handles=[handle], loc="lower center", bbox_to_anchor=(0.84, 0.065), handletextpad=0.5)

    # Save the figure and its values
    fig.subplots_adjust(left=0.13, right=0.99, top=0.93, bottom=0.42, wspace=0.08)
    c.save(fig, stem)
    c.write_values(stem, pd.DataFrame(rows))

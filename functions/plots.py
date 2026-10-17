import numpy as np
from scipy.stats import gaussian_kde
from sklearn.metrics import roc_curve

# Render straight to file, since the cluster has no display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Colours for cases and controls
CASE, CONTROL = "tab:red", "tab:blue"


def distributions(scores, is_case, title, path):
    """Score distributions, cases over controls."""

    # Draw a smoothed curve over the range of scores, for cases and for controls
    plt.figure(figsize=(7, 4.5))
    grid = np.linspace(scores.min(), scores.max(), 400)
    for group, colour, name in [(0, CONTROL, "Control"), (1, CASE, "Case")]:
        data = np.asarray(scores[is_case == group])
        curve = gaussian_kde(data)(grid)
        plt.plot(grid, curve, color=colour, label=f"{name} (n={len(data)})")
        plt.fill_between(grid, curve, color=colour, alpha=0.25)

    # Label the axes and write the PNG
    plt.xlabel("Score"); plt.ylabel("Density"); plt.title(title); plt.legend()
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def confusion_table(y, score, title, path):
    """2x2 counts at the Youden-J threshold, with the rates underneath."""

    # Take the threshold that maximises sensitivity + specificity
    y, score = np.asarray(y), np.asarray(score, dtype=float)
    fpr, tpr, thresholds = roc_curve(y, score)
    predicted_case = score >= thresholds[np.argmax(tpr - fpr)]

    # Count how many cases and controls fall either side of that threshold
    tp = int(((y == 1) & predicted_case).sum())
    fn = int(((y == 1) & ~predicted_case).sum())
    fp = int(((y == 0) & predicted_case).sum())
    tn = int(((y == 0) & ~predicted_case).sum())
    rate = lambda num, den: num / den if den else float("nan")

    # Lay the four counts out as a 2x2 table
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.axis("off"); ax.set_title(title, pad=12)
    table = ax.table(cellText=[["", "Pred case", "Pred control"],
                               ["Actual case", tp, fn],
                               ["Actual control", fp, tn]],
                     loc="center", cellLoc="center")
    table.scale(1, 2.2)
    for (row, col), cell in table.get_celld().items():
        if row == 0 or col == 0:
            cell.set_text_props(weight="bold")

    # Put sensitivity, specificity, PPV and NPV underneath, and write the PNG
    ax.text(0.5, -0.05,
            f"Sensitivity {rate(tp, tp + fn):.2f}   Specificity {rate(tn, tn + fp):.2f}   "
            f"PPV {rate(tp, tp + fp):.2f}   NPV {rate(tn, tn + fn):.2f}",
            ha="center", va="top", transform=ax.transAxes)
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()

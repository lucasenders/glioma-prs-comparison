"""Mean AUC with Nadeau-Bengio intervals and the paired comparisons, over every repeat in paper_results/. Output goes to summary/out/.

Run with: python run.py [--results <paper_results folder>] [--out <folder>]
"""

import argparse
import glob
import os
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS = os.path.join(os.path.dirname(HERE), "paper_results")

# Ratio of test to training set size in five-fold cross-validation
TEST_TO_TRAIN = 1 / 4

# Metrics, the reference methods, and the model left out of the output
METRICS = ["test_auc", "age_40_64_auc", "train_auc"]
PUBLISHED = "melin_25"
LINEAR = "elastic_net"
NON_LINEAR = ["random_forest", "xgboost_model", "mlp", "mlp_narrow", "mlp_shallow", "mlp_small"]
NOT_REPORTED = ["logistic"]

# Output names of the two fixed sets
LABELS = {"panel": "25 variant set", "known50": "50 variant set"}

# Order of the datasets, feature sets and approaches in the tables
DATASETS = ["AllGlioma", "GBM", "nonGBM"]
FEATURE_ORDER = ["25 variant set", "50 variant set", "threshold_1e-05", "threshold_0.0001",
                 "threshold_0.001", "ct_1e-05", "ct_0.0001", "ct_0.001"]
ARM_ORDER = ["published_prs", "genome_wide_prs", "standard_prs", "ml_prs", "dl_prs"]

def read_folds(results):
    """Read every per-fold AUC: repeat, approach, model, feature set, dataset, fold, metric and AUC."""

    rows = []
    for path in sorted(glob.glob(f"{results}/repeat*/*/**/test_auc.csv", recursive=True)):
        # Read the repeat, approach, model and feature set from the folder names
        folder = os.path.dirname(path)
        parts = os.path.relpath(folder, results).split(os.sep)
        repeat, arm = int(parts[0][len("repeat"):]), parts[1]
        if arm == "published_prs":
            model, feature_set = "published", parts[2]
        elif arm == "genome_wide_prs":
            model, feature_set = parts[2], "genome_wide"
        elif arm == "standard_prs":
            model, feature_set = "standard_prs", parts[3]
        else:
            model, feature_set = parts[2], parts[3]

        # Read each metric's per-fold AUC
        for metric in METRICS:
            file = f"{folder}/{metric}.csv"
            if not os.path.exists(file):
                continue
            table = pd.read_csv(file)
            table = table.assign(repeat=repeat, arm=arm, model=model, feature_set=feature_set,
                                 metric=metric, result=repeat * 100 + table["fold"])
            rows.append(table[["repeat", "arm", "model", "feature_set", "dataset", "fold",
                               "result", "metric", "auc"]])

    # Rename the fixed sets and drop the model that is not reported
    folds = pd.concat(rows, ignore_index=True)
    folds["feature_set"] = folds["feature_set"].replace(LABELS)
    return folds[~folds["model"].isin(NOT_REPORTED)].reset_index(drop=True)

def interval(values):
    """Return the mean of the fold results and its Nadeau-Bengio 95% interval."""

    values = np.asarray(values, dtype=float)
    j = len(values)
    if j < 2:
        raise ValueError(f"{j} fold results, cannot form an interval")
    mean = values.mean()
    half = stats.t.ppf(0.975, j - 1) * values.std(ddof=1) * np.sqrt(1 / j + TEST_TO_TRAIN)
    return mean, mean - half, mean + half, j

def summarise(folds):
    """Summarise each approach, model, feature set, dataset and metric, with the fold values beside."""

    rows = []
    keys = ["arm", "model", "feature_set", "dataset", "metric"]
    for key, part in folds.groupby(keys, sort=False):
        part = part.sort_values("result")
        mean, lo, hi, j = interval(part["auc"])
        row = dict(zip(keys, key))
        row.update(J=j, mean=mean, lo=max(lo, 0), hi=min(hi, 1), sd=part["auc"].std(ddof=1))
        row.update({f"r{r}f{k}": v for r, k, v in zip(part["repeat"], part["fold"], part["auc"])})
        rows.append(row)
    return pd.DataFrame(rows)

def compare(folds, pairs, label):
    """Compute the paired AUC differences, A minus B, within each fold, for each (A, B) pair, dataset and metric."""

    rows = []
    keyed = folds.set_index(["arm", "model", "feature_set", "dataset", "metric", "result"])["auc"].sort_index()
    for a, b in pairs:
        for dataset in DATASETS:
            for metric in ["test_auc", "age_40_64_auc"]:
                # Line up the two methods' fold results, skipping a pair with any fold missing
                try:
                    x = keyed.loc[(*a, dataset, metric)].sort_index()
                    y = keyed.loc[(*b, dataset, metric)].sort_index()
                except KeyError:
                    continue
                y = y.reindex(x.index)
                if y.isna().any():
                    continue

                # Average the differences and form the interval
                diff = (x - y).values
                mean, lo, hi, j = interval(diff)
                rows.append({"comparison": label, "dataset": dataset, "metric": metric,
                             "arm": a[0], "model": a[1], "feature_set": a[2],
                             "reference_model": b[1], "reference_feature_set": b[2], "J": j,
                             "mean_auc": x.mean(), "reference_mean_auc": y.mean(),
                             "difference": mean, "lo": lo, "hi": hi,
                             "folds_better": int((diff > 0).sum())})
    return pd.DataFrame(rows)

def fmt(row):
    """Format a mean and its interval to three decimals."""

    return f"{row['mean']:.3f} ({row['lo']:.3f}-{row['hi']:.3f})"

def markdown(summary, results):
    """Write one table per dataset, with every method down the side and the three metrics across."""

    # Write the heading
    j = sorted(summary["J"].unique())
    lines = ["# AUC by method, mean of the fold results with Nadeau-Bengio 95% intervals", "",
             f"Generated from `{results}`. J = {j}. Interval: mean +/- t(0.975, J-1) x sd x "
             f"sqrt(1/J + {TEST_TO_TRAIN}); limits kept within [0, 1].", ""]

    # Sort the rows by approach and feature set
    summary = summary.copy()
    summary["arm"] = pd.Categorical(summary["arm"], ARM_ORDER, ordered=True)
    extra = sorted(set(summary["feature_set"]) - set(FEATURE_ORDER))
    summary["feature_set"] = pd.Categorical(summary["feature_set"], FEATURE_ORDER + extra, ordered=True)

    # Write each dataset's table
    for dataset in DATASETS:
        lines += [f"## {dataset}", "", "| Arm | Model | Feature set | Test AUC | Age 40-64 AUC | Training AUC |",
                  "|---|---|---|---|---|---|"]
        part = summary[summary["dataset"] == dataset]
        for (arm, model, fs), cell in part.groupby(["arm", "model", "feature_set"], sort=True, observed=True):
            by = cell.set_index("metric")
            cells = [fmt(by.loc[m]) if m in by.index else "-" for m in METRICS]
            lines.append(f"| {arm} | {model} | {fs} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)

def main():
    """Write the summary, the AUC tables and the three comparison files."""

    # Read the arguments
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default=DEFAULT_RESULTS)
    parser.add_argument("--out", default=os.path.join(HERE, "out"))
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # Write the summary and the AUC tables
    folds = read_folds(args.results)
    print(f"{folds['repeat'].nunique()} repeat(s), {folds['result'].nunique()} fold results")
    summary = summarise(folds)
    summary.to_csv(f"{args.out}/auc_summary.csv", index=False)
    with open(f"{args.out}/auc_tables.md", "w") as handle:
        handle.write(markdown(summary, args.results))

    # Compare every method with the Melin score
    methods = folds[["arm", "model", "feature_set"]].drop_duplicates()
    ref = ("published_prs", "published", PUBLISHED)
    pairs = [(tuple(m), ref) for m in methods.itertuples(index=False) if tuple(m) != ref]
    compare(folds, pairs, f"vs {PUBLISHED}").to_csv(
        f"{args.out}/compare_vs_published.csv", index=False)

    # Compare each ML and DL model with the standard PRS on the same feature set
    fitted = methods[methods["arm"].isin(["ml_prs", "dl_prs"])]
    pairs = [(tuple(m), ("standard_prs", "standard_prs", m.feature_set)) for m in fitted.itertuples(index=False)]
    compare(folds, pairs, "vs standard PRS, same feature set").to_csv(
        f"{args.out}/compare_vs_standard.csv", index=False)

    # Compare each non-linear model with elastic net on the same feature set
    pairs = [(tuple(m), ("ml_prs", LINEAR, m.feature_set))
             for m in fitted[fitted["model"].isin(NON_LINEAR)].itertuples(index=False)]
    compare(folds, pairs, "vs elastic net, same feature set").to_csv(
        f"{args.out}/compare_vs_linear.csv", index=False)
    print(f"{len(summary)} summary rows; tables in {args.out}/")

if __name__ == "__main__":
    main()

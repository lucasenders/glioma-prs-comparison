"""Number of final SVM fits that chose the linear and the radial basis function kernel, per dataset and feature set. Output goes to summary/out/svm_kernels.csv.

Run with: python svm_kernels.py [--out <folder>]        after every repeat's ML jobs have finished
"""

import argparse
import glob
import os
import sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import config as cfg

def read_chosen():
    """Read the chosen kernel of every final SVM fit, with its repeat, dataset, fold and feature set."""

    rows = []
    for path in glob.glob(os.path.join(cfg.OUT, "repeat*", "*", "fold*", "ml", "svm", "*", "chosen.csv")):
        parts = os.path.relpath(path, cfg.OUT).split(os.sep)
        kernel = pd.read_csv(path)["kernel"].iloc[0]
        rows.append({"repeat": parts[0], "dataset": parts[1], "fold": parts[2], "feature_set": parts[5], "kernel": kernel})
    return pd.DataFrame(rows)

if __name__ == "__main__":
    # Read the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join(HERE, "out"))
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # Read every final SVM fit and check none is missing
    fits = read_chosen()
    expected = fits["repeat"].nunique() * len(cfg.DATASETS) * cfg.N_FOLDS * len(cfg.SETS)
    if len(fits) != expected:
        raise ValueError(f"{len(fits)} SVM fits found, expected {expected}")

    # Count each kernel per dataset and feature set, and write svm_kernels.csv
    counts = fits.groupby(["dataset", "feature_set", "kernel"]).size().rename("n").reset_index()
    counts["of"] = counts.groupby(["dataset", "feature_set"])["n"].transform("sum")
    counts.to_csv(f"{args.out}/svm_kernels.csv", index=False)

    # Report the counts
    print(f"{len(fits)} SVM fits over {fits['repeat'].nunique()} repeat(s)")
    print("all datasets:", fits["kernel"].value_counts().to_dict())
    for dataset, part in fits.groupby("dataset"):
        print(f"{dataset}:", part["kernel"].value_counts().to_dict(), f"of {len(part)}")
    print(f"\n>>> DONE: {args.out}/svm_kernels.csv")

"""Appendix G: number of final SVM fits that chose each kernel, per dataset and feature set, from pipeline/summary/out/svm_kernels.csv. Output goes to appendix_G.txt beside this script.

Run with: python3 appendix_g.py [--summary <folder>] [--out <folder>]
"""

import argparse
import os
import pandas as pd

# Folder paths
HERE = os.path.dirname(os.path.abspath(__file__))
SUMMARY = os.path.join(os.path.dirname(os.path.dirname(HERE)), "pipeline", "summary", "out")

# Datasets, feature sets and kernels, with their display names
DATASETS = [("AllGlioma", "All glioma"), ("GBM", "GBM"), ("nonGBM", "Non-GBM")]
SETS = [("panel", "25-variant set"), ("known50", "50-variant set"),
        ("ct_1e-05", "C+T 1×10−5"), ("threshold_1e-05", "Threshold 1×10−5"),
        ("ct_0.0001", "C+T 1×10−4"), ("threshold_0.0001", "Threshold 1×10−4"),
        ("ct_0.001", "C+T 1×10−3"), ("threshold_0.001", "Threshold 1×10−3")]
KERNELS = [("linear", "linear"), ("rbf", "RBF")]

if __name__ == "__main__":
    # Read the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default=SUMMARY)
    parser.add_argument("--out", default=HERE)
    args = parser.parse_args()

    # Read the kernel counts
    counts = pd.read_csv(os.path.join(args.summary, "svm_kernels.csv"))
    n = counts.set_index(["dataset", "feature_set", "kernel"])["n"]
    get = lambda d, s, k: int(n.get((d, s, k), 0))

    # Write one row per feature set, then the totals
    out = ["Table G1. Kernel chosen by the SVM in its final fits, linear or radial basis function (RBF), of 20 fits per cell."]
    out.append("\t".join(["Feature set"] + [f"{name} {label}" for _, name in DATASETS for _, label in KERNELS]))
    for s, label in SETS:
        out.append("\t".join([label] + [str(get(d, s, k)) for d, _ in DATASETS for k, _ in KERNELS]))
    out.append("\t".join(["Total (of 160)"] + [str(sum(get(d, s, k) for s, _ in SETS)) for d, _ in DATASETS for k, _ in KERNELS]))
    total = {k: sum(get(d, s, k) for d, _ in DATASETS for s, _ in SETS) for k, _ in KERNELS}
    out.append(f"Over all 480 fits: linear {total['linear']}, RBF {total['rbf']}.")

    # Save appendix_G.txt
    with open(os.path.join(args.out, "appendix_G.txt"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(out))
    print("\n".join(out))

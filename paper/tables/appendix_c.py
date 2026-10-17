"""Appendix C: mean test AUC (95% CI) of every method on every feature set, copied from pipeline/summary/out/auc_tables.md. Output goes to appendix_C.txt beside this script.

Run with: python3 appendix_c.py
"""

import os
import re

# Folder paths
HERE = os.path.dirname(os.path.abspath(__file__))
MD = os.path.join(os.path.dirname(os.path.dirname(HERE)), "pipeline", "summary", "out", "auc_tables.md")

# Datasets, methods, feature sets and the methods without feature sets, with their display names
DATASETS = [("AllGlioma", "all glioma"), ("GBM", "GBM"), ("nonGBM", "non-GBM")]
ROWS = [("standard_prs", "Standard PRS"), ("elastic_net", "Elastic net"), ("random_forest", "Random forest"),
        ("xgboost_model", "XGBoost"), ("svm", "SVM"), ("mlp", "MLP 256-64-16"), ("mlp_narrow", "MLP 64-16-4"),
        ("mlp_shallow", "MLP 256-64"), ("mlp_small", "MLP 64-16")]
SETS = [("25 variant set", "25-variant set"), ("50 variant set", "50-variant set"),
        ("ct_1e-05", "C+T 1×10−5"), ("threshold_1e-05", "Threshold 1×10−5"),
        ("ct_0.0001", "C+T 1×10−4"), ("threshold_0.0001", "Threshold 1×10−4"),
        ("ct_0.001", "C+T 1×10−3"), ("threshold_0.001", "Threshold 1×10−3")]
OTHERS = [("published", "melin_25", "Melin"), ("published", "eckel_passow_25", "Eckel-Passow"),
          ("lassosum", "genome_wide", "lassosum"), ("ldpred2", "genome_wide", "LDpred2"), ("prs_cs", "genome_wide", "PRS-CS")]

# Format of an AUC cell
CELL = re.compile(r"^\d\.\d{3} \(\d\.\d{3}-\d\.\d{3}\)$")


def read_md():
    """Read the test AUC cell of every dataset, model and feature set in auc_tables.md, exactly as printed."""

    cells, dataset = {}, None
    for line in open(MD, encoding="utf-8"):
        if line.startswith("## "):
            dataset = line[3:].strip()
        elif line.startswith("| ") and dataset and not line.startswith("| Arm") and not line.startswith("|---"):
            parts = [p.strip() for p in line.strip().strip("|").split("|")]
            assert CELL.match(parts[3]), parts
            cells[(dataset, parts[1], parts[2])] = parts[3].replace("-", "–")
    return cells


# Write one table per dataset, then a line for the methods without feature sets
cells = read_md()
out = []
for i, (key, name) in enumerate(DATASETS, 1):
    out.append(f"Table C{i}. Mean test AUC (95% CI) for {name}.")
    out.append("\t".join(["Method"] + [label for _, label in SETS]))
    for model, label in ROWS:
        out.append("\t".join([label] + [cells[(key, model, s)] for s, _ in SETS]))
    others = [f"{label} {cells[(key, model, fset)]}" for model, fset, label in OTHERS if (key, model, fset) in cells]
    out.append("Methods without feature sets: " + "; ".join(others) + ".")
    out.append("")

# Save appendix_C.txt and report the number of cells copied
text = "\n".join(out)
with open(os.path.join(HERE, "appendix_C.txt"), "w", encoding="utf-8") as handle:
    handle.write(text)
print(text)
print(f"{len(ROWS) * len(SETS) * len(DATASETS)} table cells and {sum(1 for d, _ in DATASETS for m, f, _ in OTHERS if (d, m, f) in cells)} other cells copied from auc_tables.md")

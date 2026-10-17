"""Appendices D, E, F, H and I: the paired comparisons, training AUC and gap, and age 40 to 64 AUC, from pipeline/summary/out/. Output goes to appendix_D.txt to appendix_I.txt beside this script.

Run with: python3 appendix_d_to_i.py [--summary <folder>] [--out <folder>]
"""

import argparse
import os
import re
from decimal import Decimal, ROUND_HALF_UP
import pandas as pd

# Folder paths
HERE = os.path.dirname(os.path.abspath(__file__))
SUMMARY = os.path.join(os.path.dirname(os.path.dirname(HERE)), "pipeline", "summary", "out")

# Datasets, methods, feature sets and the methods without feature sets, with their display names
DATASETS = [("AllGlioma", "all glioma"), ("GBM", "GBM"), ("nonGBM", "non-GBM")]
MODELS = [("standard_prs", "Standard PRS"), ("elastic_net", "Elastic net"), ("random_forest", "Random forest"),
          ("xgboost_model", "XGBoost"), ("svm", "SVM"), ("mlp", "MLP 256-64-16"), ("mlp_narrow", "MLP 64-16-4"),
          ("mlp_shallow", "MLP 256-64"), ("mlp_small", "MLP 64-16")]
SETS = [("25 variant set", "25-variant set"), ("50 variant set", "50-variant set"),
        ("ct_1e-05", "C+T 1×10−5"), ("threshold_1e-05", "Threshold 1×10−5"),
        ("ct_0.0001", "C+T 1×10−4"), ("threshold_0.0001", "Threshold 1×10−4"),
        ("ct_0.001", "C+T 1×10−3"), ("threshold_0.001", "Threshold 1×10−3")]
OTHERS = [("published", "melin_25", "Melin"), ("published", "eckel_passow_25", "Eckel-Passow"),
          ("lassosum", "genome_wide", "lassosum"), ("ldpred2", "genome_wide", "LDpred2"),
          ("prs_cs", "genome_wide", "PRS-CS")]

# Format of an AUC cell, and the column of each metric in auc_tables.md
CELL = re.compile(r"^\d\.\d{3} \(\d\.\d{3}-\d\.\d{3}\)$")
COLUMN = {"test": 3, "age": 4, "train": 5}

def three(x):
    """Round a value half up to three decimals, with a true minus sign."""

    text = str(Decimal(str(x)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))
    return text.replace("-", "−")

def read_md(path):
    """Read every AUC cell of auc_tables.md, keyed by dataset, model, feature set and metric, exactly as printed."""

    cells, dataset = {}, None
    for line in open(path, encoding="utf-8"):
        if line.startswith("## "):
            dataset = line[3:].strip()
        elif line.startswith("| ") and dataset and not line.startswith("| Arm") and not line.startswith("|---"):
            parts = [p.strip() for p in line.strip().strip("|").split("|")]
            for name, i in COLUMN.items():
                if parts[i] != "-":
                    assert CELL.match(parts[i]), parts
                    cells[(dataset, parts[1], parts[2], name)] = parts[i].replace("-", "–")
    return cells

def auc_table(cells, column, letter, title, models, others):
    """Write one appendix of AUC cells, a table per dataset captioned Table <letter>1 to 3."""

    out = []
    for i, (key, name) in enumerate(DATASETS, 1):
        out.append(f"Table {letter}{i}. {title.format(name)}.")
        out.append("\t".join(["Method"] + [label for _, label in SETS]))
        for model, label in models:
            out.append("\t".join([label] + [cells[(key, model, s, column)] for s, _ in SETS]))
        rest = [f"{label} {cells[(key, model, fset, column)]}" for model, fset, label in others
                if (key, model, fset, column) in cells]
        out.append("Methods without feature sets: " + "; ".join(rest) + ".")
        out.append("")
    return out

def difference_table(table, letter, title, models, others=()):
    """Write one appendix of paired differences, a table per dataset captioned Table <letter>1 to 3."""

    # Format each test AUC difference and its CI
    table = table[table["metric"] == "test_auc"]
    cell = {(r.dataset, r.model, r.feature_set): f"{three(r.difference)} ({three(r.lo)} to {three(r.hi)})"
            for r in table.itertuples()}

    # Write each dataset's table, with a note wherever a limit rounds to −0.000
    out = []
    for i, (key, name) in enumerate(DATASETS, 1):
        start = len(out)
        out.append(f"Table {letter}{i}. {title.format(name)}.")
        out.append("\t".join(["Method"] + [label for _, label in SETS]))
        for model, label in models:
            out.append("\t".join([label] + [cell[(key, model, s)] for s, _ in SETS]))
        if others:
            rest = [f"{label} {cell[(key, model, fset)]}" for model, fset, label in others if (key, model, fset) in cell]
            out.append("Methods without feature sets: " + "; ".join(rest) + ".")
        if any("−0.000" in line for line in out[start:]):
            out.append("Note. −0.000 marks a limit below zero that rounds to zero, so that CI lies just below zero.")
        out.append("")
    return out, len(cell)

def gap_table(summary, letter):
    """Write the mean training AUC minus mean test AUC, a table per dataset captioned Table <letter>4 to 6."""

    # Compute each gap from the full-precision means
    means = summary.pivot_table(index=["dataset", "model", "feature_set"], columns="metric", values="mean")
    gap = (means["train_auc"] - means["test_auc"]).dropna()

    # Write each dataset's table
    out = []
    for i, (key, name) in enumerate(DATASETS, 4):
        out.append(f"Table {letter}{i}. Gap between mean training AUC and mean test AUC for {name}.")
        out.append("\t".join(["Method"] + [label for _, label in SETS]))
        for model, label in MODELS:
            out.append("\t".join([label] + [three(gap[(key, model, s)]) for s, _ in SETS]))
        rest = [f"{label} {three(gap[(key, model, fset)])}" for model, fset, label in OTHERS
                if (key, model, fset) in gap.index]
        out.append("Methods without feature sets: " + "; ".join(rest) + ".")
        out.append("")
    return out

if __name__ == "__main__":
    # Read the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default=SUMMARY)
    parser.add_argument("--out", default=HERE)
    args = parser.parse_args()

    # Read the summary output
    cells = read_md(os.path.join(args.summary, "auc_tables.md"))
    summary = pd.read_csv(os.path.join(args.summary, "auc_summary.csv"))
    compare = {n: pd.read_csv(os.path.join(args.summary, f"compare_vs_{n}.csv"))
               for n in ["published", "standard", "linear"]}
    fitted = [m for m in MODELS if m[0] != "standard_prs"]
    nonlinear = [m for m in fitted if m[0] not in ("elastic_net", "svm")]

    # Build the five appendices
    tables = {}
    tables["D"], n_d = difference_table(compare["published"], "D", "Paired difference in mean test AUC from the Melin score "
                                        "(95% CI) for {}", MODELS, OTHERS[1:])
    tables["E"], n_e = difference_table(compare["standard"], "E", "Paired difference in mean test AUC from the standard PRS "
                                        "on the same feature set (95% CI) for {}", fitted)
    tables["F"], n_f = difference_table(compare["linear"], "F", "Paired difference in mean test AUC from elastic net "
                                        "on the same feature set (95% CI) for {}", nonlinear)
    tables["H"] = auc_table(cells, "train", "H", "Mean training AUC (95% CI) for {}", MODELS, OTHERS[2:]) + gap_table(summary, "H")
    tables["I"] = auc_table(cells, "age", "I", "Mean test AUC (95% CI) for {} in participants aged 40 to 64", MODELS, OTHERS)

    # Save each appendix and report the number of differences
    for letter, lines in tables.items():
        with open(os.path.join(args.out, f"appendix_{letter}.txt"), "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines))
    print(f"D {n_d} differences from Melin, E {n_e} from the standard PRS, F {n_f} from elastic net")
    print(f">>> DONE: appendix_D, E, F, H and I.txt in {args.out}")

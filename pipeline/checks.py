"""Checks of one fold: ID separation, fixed-set variants with no weight, and runtimes. Output goes to out/repeat<r>/<dataset>/fold<k>/checks.md.

Run with: python checks.py <dataset> <k> [--repeat r]        after collect/run.py for that fold
"""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg

def separation(repeat, dataset, k, lines):
    """Check no test person is in any training set, and no inner validation person in its inner training set."""

    # Check the outer training and test sets
    out_dir = cfg.outer_dir(repeat, dataset, k)
    train, test = set(cfg.read_ids(f"{out_dir}/train.txt")), set(cfg.read_ids(f"{out_dir}/test.txt"))
    problems = []
    if train & test:
        problems.append(f"{len(train & test)} people in both train.txt and test.txt")

    # Check each inner fold's training and validation sets
    for j in range(1, cfg.INNER + 1):
        inner = cfg.inner_dir(repeat, dataset, k, j)
        fit, val = set(cfg.read_ids(f"{inner}/train.txt")), set(cfg.read_ids(f"{inner}/val.txt"))
        if fit & val:
            problems.append(f"inner{j}: {len(fit & val)} people in both halves")
        if (fit | val) != train:
            problems.append(f"inner{j}: the two halves are not the outer training set")
        if val & test or fit & test:
            problems.append(f"inner{j}: a test person inside the inner fold")

        # Check the inner GWAS and clumping logs keep the inner training set
        for log, tool in [(f"{inner}/gwas.log", "plink2"), (f"{inner}/clump.log", "plink")]:
            text = open(log).read()
            if f"--keep {inner}/train.txt" not in text:
                problems.append(f"inner{j}: {os.path.basename(log)} does not keep inner{j}/train.txt")
            if tool == "plink2" and f"{len(fit)} samples" not in text and f"{len(fit)} people" not in text:
                problems.append(f"inner{j}: gwas.log does not report {len(fit)} samples")

    # Check the outer clumping log keeps the outer training set
    text = open(f"{cfg.fold_dir(repeat, dataset, k)}/outer_sets/clump.log").read()
    if f"--keep {out_dir}/train.txt" not in text:
        problems.append("outer clump.log does not keep the outer train.txt")

    # Check every scores file marks exactly the test set as held out
    for path in sorted(glob_scores(repeat, dataset, k)):
        scores = pd.read_csv(path, dtype={"IID": str})
        held = set(scores.loc[~scores["is_train"], "IID"])
        if held != test:
            problems.append(f"{os.path.relpath(path, cfg.OUT)}: held-out people are not the test set")

    # Report the result
    lines.append("## 1. ID separation\n")
    lines += [f"- FAIL: {p}" for p in problems] or ["- PASS: no overlap found; every log keeps the right list"]
    lines.append("")

def glob_scores(repeat, dataset, k):
    """Find every scores.csv written in this fold."""

    import glob
    root = cfg.fold_dir(repeat, dataset, k)
    return glob.glob(f"{root}/ml/*/*/scores.csv") + glob.glob(f"{root}/dl/*/*/scores.csv") + \
        glob.glob(f"{root}/standard_prs/*/scores.csv") + glob.glob(f"{root}/genome_wide/*/scores.csv")

def no_weight(repeat, dataset, k, lines):
    """List the fixed-set variants with no weight in the outer GWAS."""

    path = f"{cfg.fold_dir(repeat, dataset, k)}/fixed/no_weight.txt"
    rows = [line.split() for line in open(path) if line.strip()]
    lines.append("## 2. Fixed-set variants with no weight in the outer gwas.QC\n")
    lines += [f"- {s}: {v}" for s, v in rows] or ["- none: every variant of both sets has a weight"]
    lines.append("")

def runtimes(repeat, dataset, k, lines):
    """Report the runtime of each model on each feature set."""

    # Read every results.csv in the fold
    import glob
    parts = []
    for path in glob.glob(f"{cfg.fold_dir(repeat, dataset, k)}/*/*/results.csv"):
        parts.append(pd.read_csv(path))
    lines.append("## 3. Runtimes (seconds per model and set)\n")
    if not parts:
        lines += ["- no results.csv found", ""]
        return

    # Write the runtimes as a model-by-feature-set table
    table = pd.concat(parts).pivot_table(index="model", columns="features", values="seconds", aggfunc="first")
    table = table.reindex(columns=[c for c in cfg.SETS if c in table.columns])
    lines.append(table.to_markdown())
    lines.append("")

    # Report Threshold 1e-3 on its own
    if "threshold_0.001" in table.columns:
        lines.append("Threshold 1e-3 alone: " + ", ".join(f"{m} {int(v)}s" for m, v in table["threshold_0.001"].dropna().items()))
        lines.append(f"Threshold 1e-3 total: {int(table['threshold_0.001'].sum())}s; "
                     f"all other sets total: {int(table.drop(columns='threshold_0.001').sum().sum())}s")
    lines.append("")

if __name__ == "__main__":
    # Run the three checks and write checks.md
    repeat, dataset, k, _, _ = cfg.parse(sys.argv)
    lines = [f"# Stage 2 checks: {dataset} fold{k} repeat{repeat}\n"]
    separation(repeat, dataset, k, lines)
    no_weight(repeat, dataset, k, lines)
    runtimes(repeat, dataset, k, lines)
    text = "\n".join(lines)
    with open(f"{cfg.fold_dir(repeat, dataset, k)}/checks.md", "w") as handle:
        handle.write(text)
    print(text)

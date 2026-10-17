"""Per-fold test, training and age 40 to 64 AUC of every method in one repeat. Output goes to paper_results/repeat<r>/.

Run with: python run.py [<dataset> [<k>]] [--repeat r]      collects the folds that have finished
"""

import glob
import os
import sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import config as cfg
from functions import auc, cohort, paths

# Columns of each AUC table, the published scores and their datasets, and the baseline folder
COLUMNS = ["method", "dataset", "fold", "n_cases", "n_controls", "auc", "lo", "hi"]
PUBLISHED = {"melin_25": ["AllGlioma", "GBM", "nonGBM"], "eckel_passow_25": ["GBM", "nonGBM"]}
BASELINE = os.path.join(cfg.PROJECT, "baseline", "results")

def ages(dataset):
    """Read each person's age, keyed by IID."""

    people = cohort.load_cohort(dataset)
    return people.set_index("IID")["age"]

def fold_aucs(people, label):
    """Compute the AUC in each dataset and fold of the people given."""

    rows = []
    for (dataset, k), part in people.groupby(["dataset", "fold"]):
        value, lo, hi = auc.auc_ci(part["case"], part["score"])
        rows.append([label, dataset, k, int(part["case"].sum()), int((1 - part["case"]).sum()),
                     value, lo, hi])
    return pd.DataFrame(rows, columns=COLUMNS)

def write(out_dir, everyone, label, with_train=True):
    """Write the test, age 40 to 64 and training AUC tables and the scores of one method."""

    # Write the test AUC, the age 40 to 64 AUC and, where there is one, the training AUC
    os.makedirs(out_dir, exist_ok=True)
    test = everyone[~everyone["is_train"]]
    band = test[(test["age"] >= cfg.AGE_LOW) & (test["age"] < cfg.AGE_HIGH)]
    fold_aucs(test, label).to_csv(f"{out_dir}/test_auc.csv", index=False)
    fold_aucs(band, label).to_csv(f"{out_dir}/age_40_64_auc.csv", index=False)
    if with_train:
        fold_aucs(everyone[everyone["is_train"]], label).to_csv(f"{out_dir}/train_auc.csv", index=False)

    # Write the scores and report the mean test AUC
    everyone.to_csv(f"{out_dir}/scores.csv", index=False)
    means = fold_aucs(test, label).groupby("dataset")["auc"].mean().round(3).to_dict()
    print(f"  {os.path.relpath(out_dir, cfg.PAPER):>45}  mean test AUC {means}", flush=True)

def gather(repeat, pairs, relative):
    """Stack one method's scores.csv across the folds given, with each person's age."""

    parts = []
    for dataset, k in pairs:
        path = os.path.join(cfg.fold_dir(repeat, dataset, k), relative, "scores.csv")
        if not os.path.exists(path):
            continue
        table = pd.read_csv(path, dtype={"IID": str})
        table["age"] = table["IID"].map(ages(dataset))
        table["dataset"], table["fold"] = dataset, k
        parts.append(table[["IID", "dataset", "fold", "is_train", "case", "age", "score"]])
    return pd.concat(parts, ignore_index=True) if parts else None

def published(repeat, pairs):
    """Write the Melin and Eckel-Passow scores of each test fold, from baseline/results/."""

    for model, datasets in PUBLISHED.items():
        parts = []
        for dataset in datasets:
            # Read the baseline scores
            profile = pd.read_csv(f"{BASELINE}/{model}/scores/{dataset}.profile", sep=r"\s+",
                                  dtype={"IID": str})[["IID", "SCORESUM"]]
            people = cohort.load_cohort(dataset).merge(profile.rename(columns={"SCORESUM": "score"}), on="IID")

            # Take each fold's test set
            for d, k in pairs:
                if d != dataset:
                    continue
                test = set(cfg.read_ids(f"{cfg.outer_dir(repeat, dataset, k)}/test.txt"))
                part = people[people["IID"].isin(test)].copy()
                part["dataset"], part["fold"], part["is_train"] = dataset, k, False
                parts.append(part[["IID", "dataset", "fold", "is_train", "case", "age", "score"]])
        if parts:
            write(os.path.join(cfg.paper_dir(repeat), "published_prs", model),
                  pd.concat(parts, ignore_index=True), model, with_train=False)

def run(repeat, dataset=None, k=None):
    """Collect every method of the folds given."""

    # Collect the published scores
    pairs = cfg.jobs(dataset, k)
    paper = cfg.paper_dir(repeat)
    published(repeat, pairs)

    # Collect the standard PRS on each feature set
    for name in cfg.SETS:
        table = gather(repeat, pairs, f"standard_prs/{name}")
        if table is not None:
            write(os.path.join(paper, "standard_prs", "standard_prs", name), table, name)

    # Collect the genome-wide methods
    for method in ["ldpred2", "prs_cs", "lassosum"]:
        table = gather(repeat, pairs, f"genome_wide/{method}")
        if table is not None:
            write(os.path.join(paper, "genome_wide_prs", method, "genome_wide"), table, method)

    # Collect every ML and DL model on each feature set
    for arm, folder in [("ml_prs", "ml"), ("dl_prs", "dl")]:
        models = sorted({os.path.basename(p) for d, f in pairs
                         for p in glob.glob(os.path.join(cfg.fold_dir(repeat, d, f), folder, "*"))
                         if os.path.isdir(p)})
        for model in models:
            for name in cfg.SETS:
                table = gather(repeat, pairs, f"{folder}/{model}/{name}")
                if table is not None:
                    write(os.path.join(paper, arm, model, name), table, name)

if __name__ == "__main__":
    # Collect the folds given, or every fold
    repeat, dataset, k, _, _ = cfg.parse(sys.argv)
    run(repeat, dataset, k)
    print(f"\n>>> DONE: per-fold files in {cfg.paper_dir(repeat)}/")

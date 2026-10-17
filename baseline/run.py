#!/usr/bin/env python
"""
Published glioma PRS models applied to AGOG, using each paper's ORs. Output goes to results/.
"""

import os
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functions import cohort, evaluate, matching, paths, scoring

# Folder paths
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
PANEL = os.path.join(HERE, "models", "panel.txt")

# Each model's datasets and its age and sex odds ratios
MODELS = {
    "eckel_passow_25": {
        "datasets": ["GBM", "nonGBM"],
        "covars": {
            "GBM":    {"age_or": 1.007, "sex_or": 0.641},
            "nonGBM": {"age_or": 0.946, "sex_or": 0.881},
        },
    },
    "melin_25": {
        "datasets": ["AllGlioma", "GBM", "nonGBM"],
        "covars": None,
    },
    "melin_27": {
        "datasets": ["AllGlioma", "GBM", "nonGBM"],
        "covars": None,
    },
}

# 20-year age bands for the age-stratified analysis
AGE_BANDS = [("20-39", 20, 40), ("40-59", 40, 60), ("60-79", 60, 80)]

# Age limits for the extreme groups
CASE_MAX, CTRL_MIN = 50, 70

# Row order in the master table
ANALYSES = ["snp_only", "snp_adj", "by_sex", "by_sex_adj", "by_age", "by_age_adj", "extreme_groups"]

# Figure-title text for each analysis
LABEL = {
    "snp_only":       "SNP only",
    "snp_adj":        "SNP + age + sex",
    "by_sex":         "SNP only",
    "by_sex_adj":     "SNP + age",
    "by_age":         "SNP only",
    "by_age_adj":     "SNP + sex",
    "extreme_groups": "SNP only",
}

def prs(model, dataset):
    """Match this model's SNPs to the SNPs in the dataset and score them with plink."""

    # Match the panel to the SNPs present in the dataset by chr:pos
    scored, missing, mismatched = matching.match_panel(
        PANEL,
        f"{HERE}/models/{model}/weights_{dataset}.txt",
        f"{paths.BFILE[dataset]}.bim")

    # Print how many SNPs matched, naming any that did not
    total = len(scored) + len(missing) + len(mismatched)
    note = "".join([f"  MISSING: {', '.join(missing)}" if missing else "",
                    f"  MISMATCH: {', '.join(mismatched)}" if mismatched else ""])
    print(f"    {len(scored)}/{total} SNPs matched{note}")

    # Score each person: sum of dosage x ln(OR)
    return scoring.score(scored, dataset, f"{RESULTS}/{model}/scores/{dataset}")

def run_model_on_dataset(model, dataset):
    """Run every analysis for one model on one dataset, and return the summary rows."""
    print(f"\n  --- {model} / {dataset} ---")

    # Join each person's PRS to their age, sex and case/control status
    people = cohort.load_cohort(dataset).merge(prs(model, dataset), on="IID")

    # Convert age and sex terms from odds ratios to log-odds
    covars = MODELS[model]["covars"]
    snp = people["score"]
    if covars:
        age = people["age"] * np.log(covars[dataset]["age_or"])
        sex = people["female"] * np.log(covars[dataset]["sex_or"])

    # Split the people into the subgroups the analyses run over
    everyone = [("all", people)]
    sexes = [("male", people[people["female"] == 0]),
             ("female", people[people["female"] == 1])]
    bands = [(name, people[(people["age"] >= low) & (people["age"] < high)])
             for name, low, high in AGE_BANDS]

    # Build the extreme groups: cases under 50 against controls over 70
    extreme = [(f"cases<{CASE_MAX}_ctrls>{CTRL_MIN}",
                people[((people["case"] == 1) & (people["age"] < CASE_MAX)) |
                       ((people["case"] == 0) & (people["age"] > CTRL_MIN))])]

    # List each analysis: name, score, subgroups, figure-filename tag
    plan = [("snp_only", snp, everyone, None),
            ("by_sex",   snp, sexes,    None),
            ("by_age",   snp, bands,    None),
            ("extreme_groups", snp, extreme, "extreme")]

    # Add the _adj analyses for the models with age and sex covariates
    if covars:
        plan += [("snp_adj",    snp + age + sex, everyone, None),
                 ("by_sex_adj", snp + age,       sexes,    None),
                 ("by_age_adj", snp + sex,       bands,    None)]

    # Evaluate and produce figures and one summary row per subgroup
    return [evaluate.evaluate(subgroup, scores, f"{RESULTS}/{model}",
                              analysis, dataset, stratum, LABEL[analysis], tag)
            for analysis, scores, groups, tag in plan
            for stratum, subgroup in groups]

def main():
    """Run every model over its datasets and write the master tables."""

    # Count cases and controls, once for all models
    print("\n=== demographics ===")
    cohort.demographics(paths.BFILE, f"{RESULTS}/demographics")

    # Run each model over its datasets
    for model in MODELS:
        print(f"\n=== {model} ===")
        rows = [row for dataset in MODELS[model]["datasets"]
                for row in run_model_on_dataset(model, dataset)]

        # Write this model's summary table, replacing any earlier run
        pd.DataFrame(rows, columns=evaluate.SUMMARY_COLS).to_csv(
            f"{RESULTS}/{model}/auc_summary.csv", index=False)

    # Stack every model's summary into the master tables
    evaluate.combine(RESULTS, list(MODELS), ANALYSES)
    print(f"\n>>> done  ->  {RESULTS}/master_auc.csv")

if __name__ == "__main__":
    main()

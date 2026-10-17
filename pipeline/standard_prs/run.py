"""Standard PRS on each feature set of one outer fold, weighted by ln(OR) from the outer GWAS. Output goes to out/repeat<r>/<dataset>/fold<k>/standard_prs/.

Run with: python run.py                          submits one job per fold
          python run.py <dataset> <k>            scores one fold here
          python run.py <dataset> <k> submit     submits that one fold
          add --repeat r for a later repeat
"""

import os
import subprocess
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import config as cfg
from functions import paths

# Job resources
RESOURCES = ["--time", "02:00:00", "--mem", "16G"]

def weights(out_dir):
    """Read each SNP's effect allele and ln(OR) from the outer GWAS."""

    # Read the outer GWAS and turn each OR into ln(OR)
    stats = pd.read_csv(f"{out_dir}/gwas.QC", sep="\t", usecols=["SNP", "A1", "OR"]).set_index("SNP")
    stats["beta"] = np.log(stats["OR"])
    return stats[["A1", "beta"]]

def score(dataset, stats, snps, prefix):
    """Score everyone on one SNP list with plink, and return each person's ID and score."""

    # Write the weights out in the layout plink reads
    stats.loc[snps].reset_index().to_csv(f"{prefix}.score", sep="\t", index=False, header=False)

    # Count the effect allele at each SNP and add up dosage x ln(OR)
    subprocess.run([cfg.PLINK_BIN, "--bfile", paths.BFILE[dataset], "--score", f"{prefix}.score",
                    "1", "2", "3", "sum", "--out", prefix], check=True, stdout=subprocess.DEVNULL)
    os.remove(f"{prefix}.score")

    # Return one row per person
    profile = pd.read_csv(f"{prefix}.profile", sep=r"\s+", dtype={"IID": str})[["IID", "SCORESUM"]]
    return profile.rename(columns={"SCORESUM": "score"})

def run(repeat, dataset, k):
    """Score every feature set of one fold."""

    # Read the outer GWAS, the training set and everyone's case status
    fold_dir = cfg.fold_dir(repeat, dataset, k)
    out_dir = cfg.outer_dir(repeat, dataset, k)
    stats = weights(out_dir)
    train = set(cfg.read_ids(f"{out_dir}/train.txt"))
    people = cfg.fam(dataset)[["IID", "PHENO"]].rename(columns={"PHENO": "case"})

    # Score everyone on each feature set and mark who is in the training set
    for name in cfg.SETS:
        set_dir = os.path.join(fold_dir, "standard_prs", name)
        os.makedirs(set_dir, exist_ok=True)
        snps = [s for s in cfg.read_list(f"{fold_dir}/outer_sets/{name}.txt") if s in stats.index]
        scored = people.merge(score(dataset, stats, snps, f"{set_dir}/prs"), on="IID")
        scored.insert(1, "is_train", scored["IID"].isin(train))
        scored[["IID", "is_train", "case", "score"]].to_csv(f"{set_dir}/scores.csv", index=False)
        print(f"  {name:>16}  {len(snps):>6} SNPs scored", flush=True)

def submit(repeat, dataset, k):
    """Submit one fold as its own job."""

    # Submit the fold, with its log in the fold's standard_prs folder
    log_dir = os.path.join(cfg.fold_dir(repeat, dataset, k), "standard_prs")
    cfg.submit(f"sprs_{dataset}_fold{k}", f"{log_dir}/job.log", RESOURCES,
               f"{cfg.ANACONDA} && python {__file__} {dataset} {k} --repeat {repeat}", HERE)

if __name__ == "__main__":
    # Score one fold here, or submit one job per fold
    repeat, dataset, k, _, do_submit = cfg.parse(sys.argv)
    if dataset and k and not do_submit:
        run(repeat, dataset, k)
        print(f"\n>>> DONE: {dataset} fold{k} repeat{repeat}")
    else:
        for d, f in cfg.jobs(dataset, k):
            print(f"  {d} / fold{f}")
            submit(repeat, d, f)
        print(f"\n>>> SUBMITTED: check with squeue -u $USER")

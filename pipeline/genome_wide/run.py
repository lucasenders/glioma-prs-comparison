"""LDpred2-auto, PRS-CS and lassosum on one outer fold, from the outer GWAS. Output goes to out/repeat<r>/<dataset>/fold<k>/genome_wide/<method>/.

Run with: python run.py                                   submits one job per method and fold
          python run.py <dataset> <k> <method>            runs one method on one fold here
          python run.py <dataset> <k> <method> submit     submits that one job
          python run.py <dataset> <k> all submit          submits the three methods of that fold
          python run.py <dataset> <k> <method> finish     rebuilds scores.csv from a finished run
          add --repeat r for a later repeat
"""

import glob
import hashlib
import os
import shutil
import subprocess
import sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import config as cfg
from functions import paths

# Methods, job resources and modules
METHODS = ["ldpred2", "prs_cs", "lassosum"]
LIMITS = {"ldpred2": ("12:00:00", "32G"), "prs_cs": ("12:00:00", "16G"), "lassosum": ("12:00:00", "32G")}
MODULES = f"{cfg.ANACONDA} && {cfg.R}"

# PRS-CS global shrinkage parameter
PHI = "1e-2"

def counts(out_dir, dataset):
    """Count the training cases and controls, and return the effective sample size."""

    train = set(cfg.read_ids(f"{out_dir}/train.txt"))
    status = cfg.fam(dataset)
    status = status[status["IID"].isin(train)]["PHENO"]
    n_case, n_ctrl = int((status == 1).sum()), int((status == 0).sum())
    return n_case, n_ctrl, round(4 / (1 / n_case + 1 / n_ctrl))

# Score file each method writes, its score column and its separator
SCORE_FILES = {"ldpred2": ("ldpred2_PRS.csv", "PRS", ","),
               "prs_cs": ("prs_cs.profile", "SCORESUM", r"\s+"),
               "lassosum": ("lassosum_PRS.csv", "PRS", ",")}

def finish(method, method_dir, dataset, out_dir):
    """Rewrite a method's score file as scores.csv, with case status and the training flag."""

    # Read the scores and join them to everyone's case status
    name, score_col, sep = SCORE_FILES[method]
    table = pd.read_csv(f"{method_dir}/{name}", sep=sep, dtype={"IID": str})
    people = cfg.fam(dataset)[["IID", "PHENO"]].rename(columns={"PHENO": "case"})
    train = set(cfg.read_ids(f"{out_dir}/train.txt"))
    scored = people.merge(table[["IID", score_col]].rename(columns={score_col: "score"}), on="IID")
    if len(scored) != len(people):
        raise ValueError(f"{method}: {len(scored)} of {len(people)} people have a score")

    # Mark who is in the training set and write scores.csv
    scored.insert(1, "is_train", scored["IID"].isin(train))
    scored[["IID", "is_train", "case", "score"]].to_csv(f"{method_dir}/scores.csv", index=False)
    print(f"  scores.csv: {len(scored)} people, {(~scored['is_train']).sum()} test", flush=True)

def ldpred2(repeat, dataset, k, method_dir, out_dir):
    """Run LDpred2-auto with the UK Biobank LD reference and the training case and control counts."""

    # Run LDpred2, then delete its temporary copy of the LD matrix
    n_case, n_ctrl, _ = counts(out_dir, dataset)
    try:
        subprocess.run(["Rscript", f"{HERE}/bin/LDpred-2.R",
                        "--bed", f"{paths.BFILE[dataset]}.rsid", "--cov", paths.COVAR,
                        "--ld", f"{cfg.LD}/map.rds", "--ldref_dir", f"{cfg.LD}/ldref",
                        "--sum_stats", f"{out_dir}/gwas.rsid.QC", "--trait", "bin", "--model", "auto",
                        "--train_ids", f"{out_dir}/train.txt",
                        "--n_cases", str(n_case), "--n_controls", str(n_ctrl),
                        "--out", f"{method_dir}/ldpred2"], cwd=method_dir, check=True)
    finally:
        shutil.rmtree(f"{method_dir}/tmp-data", ignore_errors=True)
    finish("ldpred2", method_dir, dataset, out_dir)

def prs_cs(repeat, dataset, k, method_dir, out_dir):
    """Run PRS-CS with the 1000 Genomes European LD reference, and score everyone with plink."""

    _, _, n_eff = counts(out_dir, dataset)
    bfile = f"{paths.BFILE[dataset]}.rsid"

    # Write the summary statistics columns PRS-CS reads
    stats = pd.read_csv(f"{out_dir}/gwas.rsid.QC", sep="\t", dtype=str)
    stats[["SNP", "A1", "A2", "OR", "SE"]].to_csv(f"{method_dir}/sumstats.txt", sep="\t", index=False)

    # Run PRS-CS and join its per-chromosome weights
    subprocess.run(["python", f"{cfg.PRSCS}/PRScs.py", f"--ref_dir={cfg.LD}/ldblk_1kg_eur",
                    f"--bim_prefix={bfile}", f"--sst_file={method_dir}/sumstats.txt",
                    f"--n_gwas={n_eff}", f"--phi={PHI}", "--seed=100", f"--out_dir={method_dir}/"],
                   cwd=method_dir, check=True)
    with open(f"{method_dir}/prs_cs_weights.txt", "w") as handle:
        for part in sorted(glob.glob(f"{method_dir}/*_pst_eff_*.txt")):
            handle.write(open(part).read())

    # Score everyone with plink
    subprocess.run([cfg.PLINK_BIN, "--bfile", bfile, "--score", f"{method_dir}/prs_cs_weights.txt",
                    "2", "4", "6", "sum", "--out", f"{method_dir}/prs_cs"],
                   check=True, stdout=subprocess.DEVNULL)
    finish("prs_cs", method_dir, dataset, out_dir)

    # Write the md5 checksum of each PRS-CS script that ran
    with open(f"{method_dir}/prscs_md5.txt", "w") as handle:
        for name in sorted(glob.glob(f"{cfg.PRSCS}/*.py")):
            if os.path.basename(name) in ("PRScs.py", "mcmc_gtb.py", "parse_genet.py", "gigrnd.py"):
                handle.write(f"{hashlib.md5(open(name, 'rb').read()).hexdigest()}  {os.path.basename(name)}\n")

def lassosum(repeat, dataset, k, method_dir, out_dir):
    """Run lassosum with LD and pseudovalidation from the training set's genotypes."""

    subprocess.run(["Rscript", f"{HERE}/bin/lassosum.R",
                    "--bed", f"{paths.BFILE[dataset]}.rsid", "--cov", paths.COVAR,
                    "--sum_stats", f"{out_dir}/gwas.rsid.QC", "--train_ids", f"{out_dir}/train.txt",
                    "--out", f"{method_dir}/lassosum"], cwd=method_dir, check=True)
    finish("lassosum", method_dir, dataset, out_dir)

def run(repeat, dataset, k, method, only_finish=False):
    """Run one method on one fold, or with only_finish, rebuild scores.csv from its score file."""

    # Rebuild scores.csv only, or run the method
    method_dir = os.path.join(cfg.fold_dir(repeat, dataset, k), "genome_wide", method)
    os.makedirs(method_dir, exist_ok=True)
    if only_finish:
        finish(method, method_dir, dataset, cfg.outer_dir(repeat, dataset, k))
        return
    globals()[method](repeat, dataset, k, method_dir, cfg.outer_dir(repeat, dataset, k))

def submit(repeat, dataset, k, method):
    """Submit one method on one fold as its own job."""

    # Submit the job, with its log in the method's folder
    log_dir = os.path.join(cfg.fold_dir(repeat, dataset, k), "genome_wide", method)
    walltime, memory = LIMITS[method]
    cfg.submit(f"{method}_{dataset}_fold{k}", f"{log_dir}/job.log", ["--time", walltime, "--mem", memory],
               f"{MODULES} && python {__file__} {dataset} {k} {method} --repeat {repeat}", HERE)

if __name__ == "__main__":
    # Run one method on one fold here, or submit one job per method and fold
    repeat, dataset, k, rest, do_submit = cfg.parse(sys.argv)
    method = rest[0] if rest else "all"
    if dataset and k and method != "all" and not do_submit:
        run(repeat, dataset, k, method, only_finish="finish" in rest)
        print(f"\n>>> DONE: {method} {dataset} fold{k} repeat{repeat}")
    else:
        for m in (METHODS if method == "all" else [method]):
            for d, f in cfg.jobs(dataset, k):
                print(f"  {m} / {d} / fold{f}")
                submit(repeat, d, f, m)
        print(f"\n>>> SUBMITTED: check with squeue -u $USER")

"""Settings shared by every step of the pipeline: seeds, feature sets, paths and job submission."""

import os
import subprocess
import sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT)
from functions import paths  # noqa: E402

# Folder paths
OUT = os.path.join(HERE, "out")
PAPER = os.path.join(HERE, "paper_results")
DATA = os.path.join(HERE, "data")
LD = os.path.join(PROJECT, "nextflow_pipeline_prs", "data", "supplement_data", "LD")
PRSCS = os.path.join(PROJECT, "nextflow_pipeline_prs", "bin")
SNPINFO = os.path.join(DATA, "snpinfo_mult_1kg_hm3")

# Published variants: the locator file, the Melin weights and the 50 known variants
PANEL = os.path.join(PROJECT, "baseline", "models", "panel.txt")
MELIN = os.path.join(PROJECT, "baseline", "models", "melin_25")
KNOWN50 = os.path.join(DATA, "alpen_50_known_variants.csv")

# Variants discovered in AGOG, left out of the fixed sets
EXCLUDED = ["rs143893586", "rs148532268", "rs4449583", "rs72709458"]

# Number of outer and inner folds, the base seed and the datasets
N_FOLDS = 5
INNER = 3
BASE_SEED = 100
DATASETS = list(paths.BFILE)

def seed(repeat):
    """Return the seed of one repeat: 100 for repeat 1, 101 for repeat 2, and so on."""

    return BASE_SEED + repeat - 1

# P-value cutoffs of thresholding and C+T, and the clumping settings
P_GRID = [1e-5, 1e-4, 1e-3]
CLUMP_R2 = 0.1
CLUMP_KB = 250

# Feature sets: the two fixed sets, the six GWAS sets, and the sets logistic regression is fitted on
FIXED_SETS = ["panel", "known50"]
GWAS_SETS = [f"threshold_{p:g}" for p in P_GRID] + [f"ct_{p:g}" for p in P_GRID]
SETS = FIXED_SETS + GWAS_SETS
LOGISTIC_SETS = ["panel", "known50", "ct_1e-05"]

# Columns beside the SNPs in the genotype matrix
LABELS = ["FID", "IID", "PHENO", "is_train"]
COVARIATES = [f"PC{i}" for i in range(1, 8)] + ["AGE", "SEX"]

# Age band of the age-restricted analysis, lower bound inclusive, upper bound exclusive
AGE_LOW, AGE_HIGH = 40, 65

# Modules the jobs load, and the plink programs, called by path
ANACONDA = "module load anaconda/2024.02-1"
PLINK_BIN = "/usr/local/plink/1.9/plink"
PLINK2_BIN = "/usr/local/plink/2.0-alpha/plink2"
R = "module load R/4.4.0-mkl"

# Cluster settings every job asks for
PARTITION = ["--partition", "comp", "--cpus-per-task", "8", "--constraint", "r9"]

def fold_dir(repeat, dataset, k):
    """Return this repeat's folder for one outer fold."""

    return os.path.join(OUT, f"repeat{repeat}", dataset, f"fold{k}")

def outer_dir(repeat, dataset, k):
    """Return the folder holding the outer fold's train.txt, test.txt, covar.txt and gwas.QC."""

    return os.path.join(fold_dir(repeat, dataset, k), "outer")

def inner_dir(repeat, dataset, k, j):
    """Return the folder of inner fold j of one outer fold."""

    return os.path.join(fold_dir(repeat, dataset, k), f"inner{j}")

def paper_dir(repeat):
    """Return the folder collect/ writes this repeat's per-fold AUC files to."""

    return os.path.join(PAPER, f"repeat{repeat}")

def read_ids(path):
    """Read the IIDs in a plink keep-list, as strings in file order."""

    return list(pd.read_csv(path, sep=r"\s+", header=None, dtype=str)[1])

def read_list(path):
    """Read the variant names in a one-column list."""

    with open(path) as handle:
        return [line.strip() for line in handle if line.strip()]

def write_list(items, path):
    """Write a one-column list."""

    with open(path, "w") as handle:
        handle.write("".join(f"{item}\n" for item in items))

def fam(dataset):
    """Read everyone in a dataset: IID, sex and case status as 0/1, in .fam order."""

    # Read the .fam and check everyone has a case status
    table = pd.read_csv(f"{paths.BFILE[dataset]}.fam", sep=r"\s+", header=None,
                        names=["FID", "IID", "PAT", "MAT", "SEX", "PHENO"], dtype={"IID": str})
    if not table["PHENO"].isin([1, 2]).all():
        raise ValueError(f"{dataset}: someone in the .fam has no case status")

    # Recode case status from 1/2 to 0/1
    table["PHENO"] = table["PHENO"] - 1
    return table[["FID", "IID", "SEX", "PHENO"]]

def submit(name, log, resources, command, cwd):
    """Submit one command as a cluster job."""

    # Submit the job, with its log in the given file
    os.makedirs(os.path.dirname(log), exist_ok=True)
    subprocess.run(["sbatch", "--job-name", name, "--output", log, *PARTITION, *resources,
                    "--wrap", command], cwd=cwd, check=True)

def parse(argv, extra=()):
    """Read `[dataset] [fold] [extra...] [submit] [--repeat r]` from the command line."""

    # Read the repeat, which defaults to 1
    args = list(argv[1:])
    repeat = 1
    if "--repeat" in args:
        at = args.index("--repeat")
        repeat = int(args[at + 1])
        del args[at:at + 2]

    # Read whether to submit, then the dataset, the fold and any extra arguments
    do_submit = "submit" in args
    if do_submit:
        args.remove("submit")
    dataset = args[0] if len(args) > 0 else None
    k = int(args[1]) if len(args) > 1 else None
    rest = args[2:]
    return repeat, dataset, k, rest, do_submit

def jobs(dataset=None, k=None):
    """Return every (dataset, fold) pair, or only the one named."""

    sets = [dataset] if dataset else DATASETS
    folds = [k] if k else range(1, N_FOLDS + 1)
    return [(d, f) for d in sets for f in folds]

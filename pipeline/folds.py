"""Outer and inner folds, every GWAS, clumping, the eight feature lists and the genotype matrix of one outer fold. Output goes to out/repeat<r>/<dataset>/fold<k>/.

Run with: python folds.py                          submits one job per outer fold
          python folds.py <dataset> <k>            builds one fold here
          python folds.py <dataset> <k> submit     submits that one fold
          add --repeat r for a later repeat
"""

import glob
import os
import subprocess
import sys
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg
from functions import matching, paths

# Job resources and modules
RESOURCES = ["--time", "04:00:00", "--mem", "32G"]
MODULES = cfg.ANACONDA

def run(command):
    """Run one plink command, keeping its output off the job log."""

    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)

def split(people, n, seed):
    """Split a table with a PHENO column into stratified folds, as (train, held-out) index pairs."""

    splitter = StratifiedKFold(n_splits=n, shuffle=True, random_state=seed)
    return list(splitter.split(people, people["PHENO"]))

def write_keep(table, path):
    """Write FID and IID with no header, the layout plink's --keep reads."""

    table[["FID", "IID"]].to_csv(path, sep="\t", index=False, header=False)

def covariates(dataset, path):
    """Write the covariate file plink2 reads: PCs and age from the covariate file, sex from the .fam."""

    # Read the PCs and age, then add sex from the .fam
    cov = pd.read_csv(paths.COVAR, sep=r"\s+", header=None)
    cov.columns = ["FID", "IID"] + [f"PC{i}" for i in range(1, cov.shape[1] - 2)] + ["AGE"]
    cov["IID"] = cov["IID"].astype(str)
    cov.merge(cfg.fam(dataset)[["IID", "SEX"]], on="IID").to_csv(path, sep="\t", index=False)

def gwas(dataset, keep, covar, prefix):
    """Run the GWAS on the people in one keep-list."""

    # Run the GWAS, adjusted for the covariates
    run([cfg.PLINK2_BIN, "--bfile", paths.BFILE[dataset], "--keep", keep, "--covar", covar,
         "--covar-variance-standardize", "--glm", "hide-covar", "--out", prefix])

    # Write the allele frequencies of the same people
    run([cfg.PLINK2_BIN, "--bfile", paths.BFILE[dataset], "--keep", keep, "--freq", "--out", prefix])

def sumstats(prefix):
    """Rewrite one GWAS as gwas.QC, with the columns every method reads."""

    # Read the GWAS and the allele frequencies
    assoc = pd.read_csv(glob.glob(f"{prefix}.*.glm.logistic*")[0], sep="\t")
    freq = pd.read_csv(f"{prefix}.afreq", sep="\t")

    # Build the summary statistics table
    out = pd.DataFrame({
        "CHR": assoc["#CHROM"], "BP": assoc["POS"], "SNP": assoc["ID"], "A1": assoc["A1"],
        "A2": assoc["REF"].where(assoc["A1"] != assoc["REF"], assoc["ALT"]),
        "N": assoc["OBS_CT"], "SE": assoc["LOG(OR)_SE"], "P": assoc["P"], "OR": assoc["OR"]})
    out["INFO"] = 1.0

    # Add the minor allele frequency and the frequency of the effect allele
    alt = out["SNP"].map(freq.set_index("ID")["ALT_FREQS"])
    out["MAF"] = pd.concat([alt, 1 - alt], axis=1).min(axis=1)
    out["A1_FREQ"] = alt.where(assoc["A1"].values == assoc["ALT"].values, 1 - alt)

    # Drop incomplete rows and write gwas.QC
    out = out.dropna()
    out.to_csv(f"{os.path.dirname(prefix)}/gwas.QC", sep="\t", index=False)
    return out

def rsid_sumstats(stats, out_dir):
    """Write gwas.rsid.QC: the SNPs in the HapMap3 reference, renamed to their rsIDs."""

    # Read the HapMap3 rsIDs, keyed by chromosome and position
    ref = pd.read_csv(cfg.SNPINFO, sep="\t", usecols=["CHR", "SNP", "BP"])
    ref["key"] = ref["CHR"].astype(str) + ":" + ref["BP"].astype(str)
    lookup = ref.drop_duplicates("key").set_index("key")["SNP"]

    # Rename each SNP to its rsID and keep only those in the reference
    stats = stats.copy()
    stats["SNP"] = (stats["CHR"].astype(str) + ":" + stats["BP"].astype(str)).map(lookup)
    stats.dropna(subset=["SNP"]).to_csv(f"{out_dir}/gwas.rsid.QC", sep="\t", index=False)

def read_stats(folder):
    """Read one GWAS's SNP and p-value, smallest p first."""

    stats = pd.read_csv(f"{folder}/gwas.QC", sep="\t", usecols=["SNP", "P"])
    return stats.dropna().sort_values("P").reset_index(drop=True)

def clump(dataset, keep, stats, prefix):
    """Clump the SNPs below the loosest cutoff, with LD from the people in the keep-list."""

    # Write the SNPs below the loosest cutoff
    cut = stats[stats["P"] < max(cfg.P_GRID)]
    cut.to_csv(f"{prefix}.assoc", sep="\t", index=False)
    cfg.write_list(cut["SNP"], f"{prefix}.extract")

    # Clump them with plink
    run([cfg.PLINK_BIN, "--bfile", paths.BFILE[dataset], "--keep", keep,
         "--extract", f"{prefix}.extract", "--clump", f"{prefix}.assoc",
         "--clump-snp-field", "SNP", "--clump-field", "P",
         "--clump-p1", f"{max(cfg.P_GRID):g}", "--clump-p2", "1",
         "--clump-r2", str(cfg.CLUMP_R2), "--clump-kb", str(cfg.CLUMP_KB), "--out", prefix])

    # Return the index SNPs and their p-values
    index = pd.read_csv(f"{prefix}.clumped", sep=r"\s+").dropna(subset=["SNP"])
    return index[["SNP", "P"]]

def gwas_sets(stats, index):
    """Build the six GWAS sets: every SNP below each cutoff, and the index SNPs below each."""

    sets = {}
    for p in cfg.P_GRID:
        sets[f"threshold_{p:g}"] = list(stats.loc[stats["P"] < p, "SNP"])
        sets[f"ct_{p:g}"] = list(index.loc[index["P"] < p, "SNP"])
    return sets

def fixed_sets(dataset, out_dir):
    """Build the 25-variant and 50-variant sets, with each variant named as in AGOG."""

    # Read the .bim, keyed by chromosome and position
    bim = pd.read_csv(f"{paths.BFILE[dataset]}.bim", sep=r"\s+", header=None,
                      names=["CHR", "SNP", "CM", "BP", "A1", "A2"])
    bim["key"] = bim["CHR"].astype(str) + ":" + bim["BP"].astype(str)
    bim = bim.drop_duplicates("key").set_index("key")

    # Match the 25 Melin variants by position and allele pair
    scored, missing, mismatched = matching.match_panel(
        cfg.PANEL, f"{cfg.MELIN}/weights_{dataset}.txt", f"{paths.BFILE[dataset]}.bim")
    if len(scored) != 25 or missing or mismatched:
        raise ValueError(f"panel: {len(scored)} matched, missing {missing}, mismatched {mismatched}")
    panel = list(scored["variant"])

    # Match the 50 known variants by position
    known = pd.read_csv(cfg.KNOWN50)
    known["key"] = known["chromosome"].astype(str) + ":" + known["position_hg19"].astype(str)
    lost = known.loc[~known["key"].isin(bim.index), "rsid"].tolist()
    if lost:
        raise ValueError(f"known50: {len(lost)} variants not in the genotypes: {lost}")
    found = bim.loc[known["key"]]

    # Check each known variant's allele pair and write the match report
    pair_ok = [{a, b} == {c, d} for a, b, c, d in zip(found["A1"], found["A2"],
                                                     known["alternative_allele"], known["risk_allele"])]
    report = known.assign(agog_name=found["SNP"].values, agog_a1=found["A1"].values,
                          agog_a2=found["A2"].values, alleles_agree=pair_ok)
    report.to_csv(f"{out_dir}/known50_match.csv", index=False)
    if not all(pair_ok):
        print(f"  known50: {sum(not x for x in pair_ok)} variants whose AGOG allele pair differs "
              f"from the CSV, see fixed/known50_match.csv", flush=True)
    known50 = list(found["SNP"])

    # Check neither set holds an AGOG-discovered variant, and that the 50 contain the 25
    names = set(known["rsid"]) | set(pd.read_csv(cfg.PANEL, sep=r"\s+")["SNP"])
    hit = [r for r in cfg.EXCLUDED if r in names]
    if hit:
        raise ValueError(f"AGOG-discovered variants in a fixed set: {hit}")
    if len(set(known50)) != 50 or not set(panel) <= set(known50):
        raise ValueError("known50 must hold 50 distinct variants including the 25")
    return {"panel": panel, "known50": known50}

def matrix(dataset, snps, fold_dir, out_dir, train_ids):
    """Write everyone's additive genotype at the SNPs listed, with the labels and covariates."""

    # Read everyone's genotypes at the listed SNPs
    listed = f"{fold_dir}/extract.txt"
    cfg.write_list(sorted(snps), listed)
    run([cfg.PLINK_BIN, "--bfile", paths.BFILE[dataset], "--extract", listed, "--recode", "A",
         "--out", f"{fold_dir}/genotypes"])
    raw = pd.read_csv(f"{fold_dir}/genotypes.raw", sep=r"\s+", dtype={"IID": str})
    raw = raw.drop(columns=["FID", "PAT", "MAT", "SEX", "PHENOTYPE"])
    raw.columns = [c if c == "IID" else c.rsplit("_", 1)[0] for c in raw.columns]

    # Add case status, the outer training flag and the covariates, and write matrix.csv
    people = cfg.fam(dataset)[["FID", "IID", "PHENO"]]
    cov = pd.read_csv(f"{out_dir}/covar.txt", sep="\t", dtype={"IID": str})
    table = people.merge(cov[["IID"] + cfg.COVARIATES], on="IID").merge(raw, on="IID")
    table.insert(3, "is_train", table["IID"].isin(train_ids))
    table.to_csv(f"{fold_dir}/matrix.csv", index=False)
    os.remove(f"{fold_dir}/genotypes.raw")
    return table

def outer(repeat, dataset, k):
    """Write the outer fold's training and test sets, covariates and GWAS."""

    # Split everyone into the outer training and test sets
    out_dir = cfg.outer_dir(repeat, dataset, k)
    os.makedirs(out_dir, exist_ok=True)
    people = cfg.fam(dataset)
    train, test = split(people, cfg.N_FOLDS, cfg.seed(repeat))[k - 1]
    write_keep(people.iloc[train], f"{out_dir}/train.txt")
    write_keep(people.iloc[test], f"{out_dir}/test.txt")

    # Run the GWAS on the training set
    covariates(dataset, f"{out_dir}/covar.txt")
    gwas(dataset, f"{out_dir}/train.txt", f"{out_dir}/covar.txt", f"{out_dir}/gwas")
    rsid_sumstats(sumstats(f"{out_dir}/gwas"), out_dir)
    return out_dir

def build(repeat, dataset, k):
    """Build everything one outer fold needs before any model is fitted."""

    # Make the outer fold and its folders
    fold_dir = cfg.fold_dir(repeat, dataset, k)
    out_dir = outer(repeat, dataset, k)
    for sub in ["outer_sets", "fixed"] + [f"inner{j}/sets" for j in range(1, cfg.INNER + 1)]:
        os.makedirs(f"{fold_dir}/{sub}", exist_ok=True)
    train_ids = cfg.read_ids(f"{out_dir}/train.txt")
    sizes = []

    # Write the fixed sets, and list their variants with no weight in the outer GWAS
    fixed = fixed_sets(dataset, f"{fold_dir}/fixed")
    weighted = set(pd.read_csv(f"{out_dir}/gwas.QC", sep="\t", usecols=["SNP"])["SNP"])
    with open(f"{fold_dir}/fixed/no_weight.txt", "w") as handle:
        for name, snps in fixed.items():
            cfg.write_list(snps, f"{fold_dir}/fixed/{name}.txt")
            for snp in snps:
                if snp not in weighted:
                    handle.write(f"{name}\t{snp}\n")

    # Write the outer feature sets, clumped on the outer training set
    stats = read_stats(out_dir)
    index = clump(dataset, f"{out_dir}/train.txt", stats, f"{fold_dir}/outer_sets/clump")
    sets = dict(fixed, **gwas_sets(stats, index))
    for name, snps in sets.items():
        cfg.write_list(snps, f"{fold_dir}/outer_sets/{name}.txt")
        sizes.append((name, "outer", len(snps)))
    print(f"  outer: " + " ".join(f"{n}={len(s)}" for n, s in sets.items()), flush=True)
    wanted = set().union(*sets.values())

    # Split the outer training set into inner folds and run a GWAS on each inner training set
    people = cfg.fam(dataset)
    people = people[people["IID"].isin(train_ids)].reset_index(drop=True)
    for j, (fit, val) in enumerate(split(people, cfg.INNER, cfg.seed(repeat)), 1):
        inner = cfg.inner_dir(repeat, dataset, k, j)
        write_keep(people.iloc[fit], f"{inner}/train.txt")
        write_keep(people.iloc[val], f"{inner}/val.txt")
        gwas(dataset, f"{inner}/train.txt", f"{out_dir}/covar.txt", f"{inner}/gwas")
        inner_stats = sumstats(f"{inner}/gwas")[["SNP", "P"]].sort_values("P").reset_index(drop=True)

        # Delete the raw plink2 output, which gwas.QC replaces
        for raw in glob.glob(f"{inner}/gwas.*.glm.logistic*") + [f"{inner}/gwas.afreq"]:
            os.remove(raw)

        # Write the inner feature sets
        inner_index = clump(dataset, f"{inner}/train.txt", inner_stats, f"{inner}/clump")
        inner_sets = dict(fixed, **gwas_sets(inner_stats, inner_index))
        for name, snps in inner_sets.items():
            cfg.write_list(snps, f"{inner}/sets/{name}.txt")
            sizes.append((name, f"inner{j}", len(snps)))
        wanted |= set().union(*inner_sets.values())
        print(f"  inner{j}: {len(fit)} fit, {len(val)} validation; "
              + " ".join(f"{n}={len(s)}" for n, s in inner_sets.items()), flush=True)

    # Write the set sizes and the genotype matrix of every SNP any set names
    pd.DataFrame(sizes, columns=["set", "level", "n_snps"]).to_csv(f"{fold_dir}/sizes.csv", index=False)
    table = matrix(dataset, wanted, fold_dir, out_dir, train_ids)
    print(f"  matrix: {len(table)} people, {len(wanted)} SNPs", flush=True)

def submit(repeat, dataset, k):
    """Submit one fold as its own job."""

    # Submit the fold, with its log in the fold's folder
    fold_dir = cfg.fold_dir(repeat, dataset, k)
    os.makedirs(fold_dir, exist_ok=True)
    cfg.submit(f"folds_{dataset}_fold{k}", f"{fold_dir}/folds.log", RESOURCES,
               f"{MODULES} && python {__file__} {dataset} {k} --repeat {repeat}", cfg.HERE)

if __name__ == "__main__":
    # Build one fold here, or submit one job per fold
    repeat, dataset, k, _, do_submit = cfg.parse(sys.argv)
    if dataset and k and not do_submit:
        build(repeat, dataset, k)
        print(f"\n>>> DONE: {dataset} fold{k} repeat{repeat}")
    else:
        for d, f in cfg.jobs(dataset, k):
            print(f"  {d} / fold{f}")
            submit(repeat, d, f)
        print(f"\n>>> SUBMITTED: check with squeue -u $USER, output lands in {cfg.OUT}/repeat{repeat}/")

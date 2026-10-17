"""Copy of each cohort restricted to the HapMap3 SNPs and named by rsID, for the genome-wide methods. Output goes beside the genotypes and to out/rsid/<dataset>/.

Run with: python make_rsid_bfiles.py        once, before genome_wide/run.py
"""

import os
import subprocess
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as cfg
from functions import paths

def lookup():
    """Read the HapMap3 rsID for each chromosome and position."""

    ref = pd.read_csv(cfg.SNPINFO, sep="\t", usecols=["CHR", "SNP", "BP"])
    ref["key"] = ref["CHR"].astype(str) + ":" + ref["BP"].astype(str)
    return ref.drop_duplicates("key").set_index("key")["SNP"]

def rename_map(bfile, rsid, out_dir):
    """Write plink's old-name to new-name table and the list of SNPs to keep."""

    # Look up each SNP's rsID by chromosome and position
    bim = pd.read_csv(f"{bfile}.bim", sep=r"\s+", header=None,
                      usecols=[0, 1, 3], names=["CHR", "SNP", "BP"])
    key = bim["CHR"].astype(str) + ":" + bim["BP"].astype(str)
    bim["rsid"] = key.map(rsid)

    # Keep only SNPs that map to a unique rsID, and write the two files
    matched = bim.dropna(subset=["rsid"]).drop_duplicates("rsid", keep=False)
    pairs = os.path.join(out_dir, "rsid_rename.txt")
    keep = os.path.join(out_dir, "rsid_keep.txt")
    matched[["SNP", "rsid"]].to_csv(pairs, sep="\t", header=False, index=False)
    matched[["SNP"]].to_csv(keep, header=False, index=False)
    print(f"  {len(matched)} of {len(bim)} variants matched the reference")
    return pairs, keep

def convert(bfile, pairs, keep):
    """Restrict the cohort to the matched SNPs and rename them to rsIDs."""

    # Restrict to the matched SNPs, then rename them
    subprocess.run([cfg.PLINK_BIN, "--bfile", bfile, "--extract", keep,
                    "--make-bed", "--out", f"{bfile}.hm3"], check=True, stdout=subprocess.DEVNULL)
    subprocess.run([cfg.PLINK_BIN, "--bfile", f"{bfile}.hm3", "--update-name", pairs, "2", "1",
                    "--make-bed", "--out", f"{bfile}.rsid"], check=True, stdout=subprocess.DEVNULL)

    # Delete the intermediate files
    for ext in ("bed", "bim", "fam", "log"):
        if os.path.exists(f"{bfile}.hm3.{ext}"):
            os.remove(f"{bfile}.hm3.{ext}")
    print(f"  wrote {bfile}.rsid with {sum(1 for _ in open(f'{bfile}.rsid.bim'))} variants")

if __name__ == "__main__":
    # Write the rsID copy of each cohort
    rsid = lookup()
    for dataset, cohort in paths.BFILE.items():
        print(f"\n=== {dataset} ===")
        out_dir = os.path.join(cfg.OUT, "rsid", dataset)
        os.makedirs(out_dir, exist_ok=True)
        convert(cohort, *rename_map(cohort, rsid, out_dir))
    print("\n>>> DONE: <cohort>.rsid beside the genotypes")

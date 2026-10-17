"""Strand check of the five palindromic variants in the 25-variant set, comparing AGOG control allele frequencies with 1000 Genomes European frequencies. Output goes to palin_check.csv beside this script.

Run with: python run.py
"""

import os
import subprocess
import sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from functions import paths  # noqa: E402

# plink path, output prefix and the largest frequency difference allowed
PLINK_BIN = "/usr/local/plink/1.9/plink"
OUT = os.path.join(HERE, "palin_check")
TOLERANCE = 0.03

# Palindromic variants and the 1000 Genomes phase 3 European frequency of the allele given (Auton et al., 2015)
VARIANTS = pd.DataFrame([
    ("rs12076373", 1, 243851947, "C", 0.163),
    ("rs12803321", 11, 118480115, "C", 0.357),
    ("rs1275600", 12, 76263551, "A", 0.405),
    ("rs77633900", 15, 76538459, "C", 0.086),
    ("rs2562152", 16, 123896, "A", 0.150),
], columns=["rsid", "CHR", "BP", "allele", "eur_freq"])

def frequencies():
    """Compute the case and control allele frequencies of the five variants in AGOG with plink."""

    # Write the five positions
    ranges = os.path.join(HERE, "palin_ranges.txt")
    with open(ranges, "w") as handle:
        for i, row in VARIANTS.iterrows():
            handle.write(f"{row.CHR} {row.BP} {row.BP} {row.rsid}\n")

    # Compute the frequencies in cases and controls
    subprocess.run([PLINK_BIN, "--bfile", paths.BFILE["AllGlioma"], "--extract", "range", ranges,
                    "--freq", "case-control", "--out", OUT], check=True, stdout=subprocess.DEVNULL)
    return pd.read_csv(f"{OUT}.frq.cc", sep=r"\s+")

def compare(frq):
    """Compare each variant's control frequency of the 1000 Genomes allele with the reference value."""

    # Match the frequencies to the five variants by position
    frq["BP"] = frq["SNP"].str.split(":").str[1].astype(int)
    table = VARIANTS.merge(frq[["CHR", "BP", "A1", "A2", "MAF_U"]], on=["CHR", "BP"], how="left")
    if table["A1"].isna().any():
        raise ValueError(f"not found in the genotypes: {list(table.loc[table['A1'].isna(), 'rsid'])}")

    # Take the control frequency of the given allele, and its difference from the reference as read and if flipped
    table["agog_freq"] = table["MAF_U"].where(table["A1"] == table["allele"], 1 - table["MAF_U"])
    table["difference"] = (table["agog_freq"] - table["eur_freq"]).abs()
    table["flipped_difference"] = (1 - table["agog_freq"] - table["eur_freq"]).abs()
    table["agrees"] = table["difference"] <= TOLERANCE
    return table[["rsid", "CHR", "BP", "allele", "agog_freq", "eur_freq", "difference",
                  "flipped_difference", "agrees"]].round(4)

if __name__ == "__main__":
    # Run the check, write palin_check.csv and report the result
    table = compare(frequencies())
    table.to_csv(f"{OUT}.csv", index=False)
    print(table.to_string(index=False))
    print(f"\nlargest difference: {table['difference'].max():.3f}")
    print(">>> ALL AGREE within 0.03" if table["agrees"].all() else ">>> CHECK: a variant differs by more than 0.03")

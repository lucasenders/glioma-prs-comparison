import os
import subprocess
import pandas as pd
from functions import paths

def score(scored, dataset, out_prefix):
    """Score everyone in the dataset, and return each person's ID and score."""

    # Write the matched weights out in the layout plink reads
    score_file = f"{out_prefix}.score"
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    scored.to_csv(score_file, sep="\t", index=False, header=False)

    # Count the paper's effect allele at each SNP and add up dosage x ln(OR)
    subprocess.run(["plink", "--bfile", paths.BFILE[dataset],
                    "--a1-allele", score_file, "2", "1",
                    "--score", score_file, "1", "2", "3", "sum",
                    "--out", out_prefix],
                   check=True, stdout=subprocess.DEVNULL)
    os.remove(score_file)

    # Hand back one row per person, ready to join to their age and sex
    profile = pd.read_csv(f"{out_prefix}.profile", sep=r"\s+")[["IID", "SCORESUM"]]
    profile["IID"] = profile["IID"].astype(str)
    return profile.rename(columns={"SCORESUM": "score"})

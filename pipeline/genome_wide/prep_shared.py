"""Convert each cohort's rsID genotypes to the bigSNP format LDpred2 reads, once per dataset. Output goes beside the genotypes, with logs in out/prep/.

Run with: python prep_shared.py        after make_rsid_bfiles.py, before genome_wide/run.py
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import config as cfg
from functions import paths

# Job resources
RESOURCES = ["--time", "02:00:00", "--mem", "32G"]

if __name__ == "__main__":
    # Submit one job per dataset, skipping any without rsID genotypes
    for dataset, bfile in paths.BFILE.items():
        bed = f"{bfile}.rsid"
        if not os.path.exists(f"{bed}.bed"):
            print(f"  {dataset:10} SKIPPED: no {bed}.bed, run make_rsid_bfiles.py first")
            continue
        cfg.submit(f"prep_{dataset}", os.path.join(cfg.OUT, "prep", f"{dataset}.log"), RESOURCES,
                   f"{cfg.R} && Rscript {HERE}/bin/prep_shared.R --bed {bed}", HERE)
        print(f"  {dataset:10} submitted")
    print("\n>>> SUBMITTED: check with squeue -u $USER")

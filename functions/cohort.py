import os
import pandas as pd
from functions import paths

# Age bands for the demographics table
AGE_BANDS = [(0, 20, "0-19"), (20, 40, "20-39"), (40, 60, "40-59"),
             (60, 80, "60-79"), (80, 200, "80+")]

def load_cohort(dataset):
    """Return one row per person: IID, age, female, case."""

    # Read sex and case/control status from the .fam
    fam = pd.read_csv(f"{paths.BFILE[dataset]}.fam", sep=r"\s+", header=None,
                      names=["FID", "IID", "PAT", "MAT", "SEX", "PHENO"])

    # Read each person's age from the last column of the covariate file
    cov = pd.read_csv(paths.COVAR, sep=r"\s+", header=None)
    cov = cov[[1, cov.columns[-1]]]
    cov.columns = ["IID", "age"]
    cov["age"] = pd.to_numeric(cov["age"], errors="coerce")

    # Put each person's age alongside their sex and status
    fam["IID"], cov["IID"] = fam["IID"].astype(str), cov["IID"].astype(str)
    people = fam.merge(cov, on="IID", how="left")

    # Drop anyone of unknown status, then record who is female and who is a case
    people = people[people["PHENO"].isin([1, 2])].copy()
    people["female"] = (people["SEX"] == 2).astype(int)
    people["case"] = (people["PHENO"] == 2).astype(int)
    return people[["IID", "age", "female", "case"]].reset_index(drop=True)

def demographics(datasets, outdir):
    """Write case/control counts by sex and 20-year age band, one CSV per dataset."""
    
    os.makedirs(outdir, exist_ok=True)
    for dataset in datasets:

        # Label each person with their sex and age band
        people = load_cohort(dataset)
        people["sex"] = people["female"].map({0: "male", 1: "female"})
        people["band"] = pd.cut(people["age"],
                                bins=[low for low, _, _ in AGE_BANDS] + [200],
                                right=False,
                                labels=[name for _, _, name in AGE_BANDS])

        # Count cases and controls in each sex and each age band
        rows = [[kind, name, int(g["case"].sum()), int((1 - g["case"]).sum()), len(g)]
                for kind, column in [("sex", "sex"), ("age", "band")]
                for name, g in people.groupby(column, observed=False)]
        pd.DataFrame(rows, columns=["group_type", "group", "cases", "controls", "total"]
                     ).to_csv(f"{outdir}/breakdown_{dataset}.csv", index=False)

        # Report the dataset's totals as it runs
        n_case = int(people["case"].sum())
        print(f"  {dataset}: {n_case} cases, {len(people) - n_case} controls")

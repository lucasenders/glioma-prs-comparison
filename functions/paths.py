import os

# The project folder, holding AGOG/ alongside baseline/ and functions/
PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGOG = os.path.join(PROJECT, "AGOG")

# Everyone's age and principal components, with IID in column 2 and age last
COVAR = os.path.join(AGOG, "covariates.txt")

# The plink .bed/.bim/.fam basename for each dataset, all sharing the controls
BFILE = {
    "AllGlioma": f"{AGOG}/AllGlioma/AllGlioma",
    "GBM":       f"{AGOG}/GBM/GBM",
    "nonGBM":    f"{AGOG}/nonGBM/nonGBM",
}

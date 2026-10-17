import numpy as np
import pandas as pd

def match_panel(panel_file, weights_file, bim_file):
    """Find this model's SNPs in the AGOG genotypes and attach their weights.

    Returns:
      scored      variant, effect_allele, beta
      missing     panel SNPs whose position is not in the .bim
      mismatched  positions found, carrying alleles the paper does not list
    """

    # Load the panel and this model's weights, keeping only the SNPs it scores
    panel = pd.read_csv(panel_file, sep=r"\s+")
    weights = pd.read_csv(weights_file, sep=r"\s+").set_index("SNP")
    panel = panel[panel["SNP"].isin(weights.index)]

    # List AGOG's variants by chromosome and position
    bim = pd.read_csv(bim_file, sep=r"\s+", header=None,
                      names=["CHR", "SNP", "CM", "BP", "A1", "A2"])
    bim["key"] = bim["CHR"].astype(str) + ":" + bim["BP"].astype(str)
    bim = bim.drop_duplicates("key").set_index("key")

    # Sort each panel SNP into scored, missing or mismatched
    scored, missing, mismatched = [], [], []
    for _, snp in panel.iterrows():

        # Set aside SNPs AGOG has no variant for at the paper's position
        key = f"{snp['CHR']}:{snp['BP']}"
        if key not in bim.index:
            missing.append(snp["SNP"])
            continue

        # Set aside SNPs where AGOG's alleles are not the pair the paper reports
        found = {bim.at[key, "A1"], bim.at[key, "A2"]}
        if found != {snp["REF"], snp["ALT"]}:
            mismatched.append(snp["SNP"])
            continue

        # Keep AGOG's variant name, the paper's effect allele, and ln(OR)
        scored.append((bim.at[key, "SNP"],
                       weights.at[snp["SNP"], "EFFECT"],
                       np.log(weights.at[snp["SNP"], "OR"])))

    return (pd.DataFrame(scored, columns=["variant", "effect_allele", "beta"]),
            missing, mismatched)

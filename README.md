# Glioma PRS comparison

Code for a Monash University thesis comparing polygenic risk scores (PRS) with machine learning (ML)
and deep learning (DL) models for glioma risk, in 560 cases and 2,237 controls from the Australian Genomics
and Clinical Outcomes of Glioma (AGOG) consortium. Three datasets are analysed (all glioma, glioblastoma and
non-glioblastoma), each against the same controls.

Every method is evaluated under one nested cross-validation framework: five outer folds, repeated four times
with different seeds (20 test folds per method), with all tuning and feature selection, including a GWAS in
each inner fold, done inside the training data. The methods are:

- **Published-weight PRS:** the 25-variant scores of Melin et al. (2017) and Eckel-Passow et al. (2019)
- **Standard PRS:** the same linear score with weights from each training fold's GWAS
- **Genome-wide PRS:** lassosum, LDpred2 (automatic mode) and PRS-CS
- **ML:** elastic net, random forest, XGBoost and a support vector machine
- **DL:** four multilayer perceptron (MLP) shapes

The ML, DL and standard PRS models are fitted on eight feature sets: the 25 published variants, 50 known
glioma variants, and variants selected from each fold's GWAS by p-value threshold or by clumping and
thresholding (C+T) at 1×10<sup>−5</sup>, 1×10<sup>−4</sup> and 1×10<sup>−3</sup>.

## Data

The AGOG genotype and phenotype data are controlled-access and are **not** included. Nothing in this
repository contains individual-level data, participant IDs or results. Access to AGOG is through the AGOG
investigators (Alpen et al., 2023).

The repository does include two small published inputs: the Melin and Eckel-Passow weights
(`baseline/models/`, odds ratios as published) and the 50 known glioma variants listed by Alpen et al. (2023)
(`pipeline/data/alpen_50_known_variants.csv`).

## Layout

| Folder | What it does |
|---|---|
| `functions/` | Shared helpers: data paths, cohort loading, variant matching, plink scoring, AUC with DeLong intervals |
| `baseline/` | Scores everyone with the published Melin and Eckel-Passow weights (`run.py`, weights in `models/`) |
| `pipeline/` | The nested cross-validation pipeline, run as SLURM jobs (details below) |
| `palindromic_check/` | Strand check of the five palindromic variants against 1000 Genomes European allele frequencies |
| `paper/figures/` | Figures 2 to 7 of the thesis |
| `paper/tables/` | Appendix tables C to I of the thesis |

Inside `pipeline/`:

| Path | What it does |
|---|---|
| `config.py` | Seeds, datasets, feature sets, tuning grids, paths and job settings shared by every step |
| `make_rsid_bfiles.py` | An rsID-named, HapMap3-restricted copy of each cohort for the genome-wide methods |
| `genome_wide/prep_shared.py` | Converts each cohort once into the bigsnpr format LDpred2 reads |
| `folds.py` | Outer and inner folds, every GWAS, clumping, the eight feature lists and the genotype matrix |
| `fitting.py` | The nested fit shared by the ML and DL arms: inner search, outer refit, test scoring |
| `ml/` | Elastic net, random forest, XGBoost and SVM (plus unpenalised logistic regression, fitted but not reported) |
| `dl/` | The four MLP shapes |
| `standard_prs/` | The standard PRS on each feature set |
| `genome_wide/` | lassosum, LDpred2 and PRS-CS (R scripts in `bin/`) |
| `collect/` | Per-fold AUC files for every method: test, training, and age 40 to 64 |
| `summary/` | Mean AUC with Nadeau-Bengio intervals, paired comparisons, and the SVM kernel counts |
| `checks.py` | Optional checks of one fold: ID separation, fixed-set variants with no weight, and runtimes |

## Setup

The code was run on Monash University's M3 cluster. To run it, the repository root needs three things that
are not included:

```
AGOG/                                   the AGOG data (layout below)
nextflow_pipeline_prs/                  Max Schuran's PRS pipeline, for PRS-CS and the LD references
    bin/PRScs.py, parse_genet.py, mcmc_gtb.py, gigrnd.py
    data/supplement_data/LD/
        map.rds, ldref/                 LDpred2's UK Biobank HapMap3 LD reference (Privé et al.)
        ldblk_1kg_eur/                  PRS-CS's 1000 Genomes European LD reference (Ge et al.)
pipeline/data/snpinfo_mult_1kg_hm3      HapMap3 variant list distributed with the PRS-CSx LD references
```

- AGOG data: `AGOG/covariates.txt` (no header; FID, IID, the 7 principal components and age), and one set of
  plink files per dataset, all sharing the same controls: `AGOG/AllGlioma/AllGlioma`, `AGOG/GBM/GBM` and
  `AGOG/nonGBM/nonGBM` (`.bed`, `.bim`, `.fam`, with case status in the `.fam`). The paths are set in
  `functions/paths.py`.
- Max's pipeline: `git clone https://github.com/xamwise/nextflow_pipeline_prs` (the analysis used commit
  `9a7bc59`). Only its PRS-CS scripts and the LD reference folder are used; Nextflow itself is not.
- PRS-CS and its reference panels: https://github.com/getian107/PRScs. The `snpinfo_mult_1kg_hm3` file comes
  with the PRS-CSx references: https://github.com/getian107/PRScsx.
- LDpred2's reference: see the LDpred2 tutorial at https://privefl.github.io/bigsnpr/articles/LDpred2.html.

The cluster settings (module names, plink paths, SLURM partition) are in `pipeline/config.py` and need
changing for any other cluster.

## Software

| Software | Version |
|---|---|
| Python | 3.11.7, packages in `requirements.txt` |
| R | 4.4.0, with bigsnpr 1.12.21, bigstatsr 1.6.2, lassosum 0.4.5, optparse, data.table, magrittr, fmsb, Matrix, bigreadr |
| PLINK 1.9 | v1.90b4.4 |
| PLINK 2 | v2.00a2LM (28 Jun 2019) |
| PRS-CS | commit 8cab5b0 |

## Run order

Every step runs from the repository root on the cluster after `module load anaconda/2024.02-1`. Steps that
submit jobs return straight away; wait for their jobs to finish before the next step.

```bash
# 1. Once: the rsID genotypes for the genome-wide methods, then their bigsnpr copies
python pipeline/make_rsid_bfiles.py
python pipeline/genome_wide/prep_shared.py

# 2. Once: the published-weight scores (calls plink from the terminal)
module load plink/1.9 && python baseline/run.py

# 3. For each repeat (replace 1 with 2, 3 and 4 in turn): folds and GWAS first, then every method,
#    then the per-fold AUCs
python pipeline/folds.py --repeat 1
python pipeline/ml/run.py --repeat 1
python pipeline/dl/run.py --repeat 1
python pipeline/standard_prs/run.py --repeat 1
python pipeline/genome_wide/run.py --repeat 1
python pipeline/collect/run.py --repeat 1

# 4. The summary over all repeats, and the SVM kernel counts
python pipeline/summary/run.py
python pipeline/summary/svm_kernels.py

# 5. Figures and appendix tables (these read pipeline/summary/out/ and run anywhere)
for n in 2 3 4 5 6 7; do (cd paper/figures && python figure$n.py); done
python paper/tables/appendix_c.py
python paper/tables/appendix_d_to_i.py
python paper/tables/appendix_g.py

# Separately: the palindromic variant check
python palindromic_check/run.py
```

Each pipeline script also runs a single dataset and fold, for example `python pipeline/folds.py GBM 1 submit`;
its first lines list the options.

## Where each result comes from

| In the thesis | Made by |
|---|---|
| Table 3 (feature set sizes) | `pipeline/folds.py`, which writes `sizes.csv` per fold |
| Table 4, Appendix C | `pipeline/summary/run.py` (`auc_tables.md`), formatted by `paper/tables/appendix_c.py` |
| Appendices D, E, F, H and I | `pipeline/summary/run.py` (`compare_vs_*.csv`, `auc_summary.csv`), formatted by `paper/tables/appendix_d_to_i.py` |
| Appendix G | `pipeline/summary/svm_kernels.py`, formatted by `paper/tables/appendix_g.py` |
| Figures 2 to 7 | `paper/figures/figure2.py` to `figure7.py` |
| Palindromic variant check (Section 3.3) | `palindromic_check/run.py` |

## Credits

- `pipeline/genome_wide/bin/LDpred-2.R` and `lassosum.R` are adapted from Max Schuran's PRS pipeline
  (https://github.com/xamwise/nextflow_pipeline_prs), released under the MIT License; its licence is in
  `pipeline/genome_wide/bin/LICENSE_nextflow_pipeline_prs`.
- PRS-CS: Ge et al. (2019), https://github.com/getian107/PRScs
- LDpred2: Privé et al. (2020), via bigsnpr
- lassosum: Mak et al. (2017)

## Licence

MIT; see `LICENSE`.

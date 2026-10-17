#!/usr/bin/env Rscript
# Adapted from the PRS pipeline of Max Schuran, https://github.com/xamwise/nextflow_pipeline_prs (commit 9a7bc59),
# released under the MIT License; see LICENSE_nextflow_pipeline_prs in this folder.

library(bigsnpr)
library(data.table)
library(magrittr)
library(optparse)

options(bigstatsr.check.parallel.blas = FALSE)
options(default.nproc.blas = NULL)

# ── Command-line options ──────────────────────────────────────────────────────

option_list <- list(
  make_option(c("-b", "--bed"),       type = "character", default = NULL,
              help = "Bed file name without extension",           metavar = "character"),
  make_option(c("--train_ids"),       type = "character", default = NULL,
              help = "file naming the training people (FID IID, no header)", metavar = "character"),
  make_option(c("-p", "--pheno"),     type = "character", default = NULL,
              help = "Phenotype file name",                       metavar = "character"),
  make_option(c("-c", "--cov"),       type = "character", default = NULL,
              help = "Covariate file name",                       metavar = "character"),
  make_option(c("-z", "--pcs"),       type = "character", default = NULL,
              help = "Principal component analysis file name",    metavar = "character"),
  make_option(c("-l", "--ld"),        type = "character", default = NULL,
              help = "LD reference map (map.rds)",                metavar = "character"),
  make_option(c("--ldref_dir"),       type = "character", default = NULL,
              help = "folder of LD_with_blocks_chr*.rds",         metavar = "character"),
  make_option(c("-f", "--sum_stats"), type = "character", default = NULL,
              help = "Summary statistics file name",              metavar = "character"),
  make_option(c("-t", "--trait"),     type = "character", default = NULL,
              help = "Trait type: 'bin' or 'quant'",             metavar = "character"),
  make_option(c("-m", "--model"),     type = "character", default = NULL,
              help = "Model type: 'inf', 'grid', or 'auto'",     metavar = "character"),
  make_option(c("--n_cases"),         type = "numeric",   default = NULL,
              help = "training-set cases, for the effective sample size", metavar = "numeric"),
  make_option(c("--n_controls"),      type = "numeric",   default = NULL,
              help = "training-set controls, for the effective sample size", metavar = "numeric"),
  make_option(c("-o", "--out"),       type = "character", default = NULL,
              help = "Output file name",                          metavar = "character")
)

opt <- parse_args(OptionParser(option_list = option_list))
set.seed(100)

if (is.null(opt$bed)) {
  stop("At least one argument must be supplied (input file).", call. = FALSE)
}

# ── Load input data ───────────────────────────────────────────────────────────

# AGOG keeps the phenotype in the .fam (column 6, 1 control, 2 case, -9
# missing) and the PCs and age in one headerless file, so fall back to those
# when no --pheno or --pcs is given.
if (is.null(opt$pheno)) {
  phenotype <- fread(paste0(opt$bed, ".fam"), header = FALSE,
                     select = c(1, 2, 6),
                     col.names = c("FID", "IID", "phenotype"))
  phenotype[!phenotype %in% c(1, 2), phenotype := NA]
  phenotype[, phenotype := phenotype - 1]
} else {
  phenotype <- fread(opt$pheno)
}

if (is.null(opt$pcs)) {
  covariate <- fread(opt$cov, header = FALSE)
  colnames(covariate) <- c("FID", "IID",
                           paste0("PC", 1:(ncol(covariate) - 3)), "AGE")
  pheno <- merge(phenotype, covariate)
} else {
  covariate <- fread(opt$cov)
  pcs       <- fread(opt$pcs)
  colnames(pcs) <- c("FID", "IID", paste0("PC", 1:(ncol(pcs) - 2)))
  pheno <- merge(phenotype, covariate) %>% merge(., pcs)
}

info     <- readRDS(opt$ld)
sumstats <- bigreadr::fread2(opt$sum_stats)

# ── Format summary statistics ─────────────────────────────────────────────────

# Read column names from the summary statistics 
stats_colnames <- colnames(sumstats)

# print the column names to check if they match the expected ones
cat(stats_colnames)

if (opt$trait == "bin") {
               
  sumstats <- sumstats[, c('CHR', 'BP', 'SNP', 'A1', 'A2', 'N', 'SE', 'P', 'OR', 'INFO', 'A1_FREQ')]

  names(sumstats) <- c("chr", "pos", "rsid", "a1", "a0",
                     "n_eff", "beta_se", "p", "OR", "INFO", "freq")

  sumstats$beta <- log(sumstats$OR)

  # Use the effective sample size for case-control data, as bigsnpr documents,
  # rather than the raw per-SNP count plink2 reports
  if (!is.null(opt$n_cases) && !is.null(opt$n_controls)) {
    sumstats$n_eff <- 4 / (1 / opt$n_cases + 1 / opt$n_controls)
    cat("n_cases =", opt$n_cases, " n_controls =", opt$n_controls,
        " n_eff =", round(sumstats$n_eff[1]), "\n")
  } else {
    stop("--n_cases and --n_controls are required for a binary trait", call. = FALSE)
  }

} else if (opt$trait == "quant") {

  sumstats <- sumstats[, c('CHR', 'BP', 'SNP', 'A1', 'A2', 'N', 'SE', 'P', 'BETA', 'INFO', 'MAF')]

  names(sumstats) <- c("chr", "pos", "rsid", "a1", "a0",
                     "n_eff", "beta_se", "p", "beta", "INFO", "MAF")

} else {
  stop("Trait type not recognised. Please specify either 'bin' or 'quant'.")
}

sumstats      <- sumstats[sumstats$rsid %in% info$rsid, ]

# ── Prepare genotype data ─────────────────────────────────────────────────────

NCORES   <- nb_cores()
# Scratch stays in the working directory, which is the job's own folder, so it is
# cleaned up with the job instead of accumulating beside the genotypes.
dir.create("tmp-data", showWarnings = FALSE)
tmp      <- tempfile(tmpdir = "tmp-data")
on.exit(file.remove(paste0(tmp, ".sbk")), add = TRUE)

# prep_shared.R converts each cohort once into <bed>_shared.bk/.rds. Without it
# every fold of every method re-encodes the same .bed, 15 times over.
shared_rds <- paste0(opt$bed, "_shared.rds")
if (file.exists(shared_rds)) {
  cat("Using the shared bigSNP at", shared_rds, "\n")
  rds_file <- shared_rds
} else {
  prefix   <- basename(opt$bed)
  rds_file <- paste0(prefix, "_LD2.rds")
  if (!file.exists(rds_file)) {
    cat("No shared bigSNP, converting BED file...\n")
    snp_readBed(paste0(opt$bed, ".bed"), backingfile = paste0(prefix, "_LD2"))
  }
}
obj.bigSNP <- snp_attach(rds_file)
genotype   <- obj.bigSNP$genotypes

map        <- obj.bigSNP$map[-3]
names(map) <- c("chr", "rsid", "pos", "a1", "a0")

# Ensure chromosome columns are compatible
sumstats$chr <- as.integer(gsub("chr", "", as.character(sumstats$chr), ignore.case = TRUE))
map$chr      <- as.integer(gsub("chr", "", as.character(map$chr),      ignore.case = TRUE))

# ── LD matrix ─────────────────────────────────────────────────────────────────
#
# Taken from the precomputed reference rather than estimated from the genotypes.
# This is what the LDpred2 authors recommend: they provide UK Biobank LD over the
# HapMap3 variants, and computing your own is meant for genotyped-only data or
# non-European samples. It also removes the cost -- snp_cor over this cohort ran
# for hours and was recomputed identically for all 15 folds.
#
# Two index spaces are needed. The variants have to be matched to the reference
# to pull the right rows out of each block, and to the genotypes to read the
# right columns when scoring. They are not the same numbering.
if (is.null(opt$ldref_dir)) {
  stop("--ldref_dir is required: the folder holding LD_with_blocks_chr*.rds",
       call. = FALSE)
}

info$chr <- as.integer(info$chr)

# Keep freq as the frequency of the matched a1 through both matches, flipping
# it whenever snp_match reverses the alleles
matched <- snp_match(sumstats, info, return_flip_and_rev = TRUE)
matched$freq <- ifelse(matched$`_REV_`, 1 - matched$freq, matched$freq)
matched$`_REV_` <- NULL; matched$`_FLIP_` <- NULL
names(matched)[names(matched) == "_NUM_ID_"] <- "ldref_id"
matched$`_NUM_ID_.ss` <- NULL

info_snp <- snp_match(matched, map, return_flip_and_rev = TRUE)
info_snp$freq <- ifelse(info_snp$`_REV_`, 1 - info_snp$freq, info_snp$freq)
info_snp$`_REV_` <- NULL; info_snp$`_FLIP_` <- NULL
cat("Kept", nrow(info_snp),
    "variants found in the summary statistics, the LD reference and the genotypes\n")

# Summary-statistic QC, as the LDpred2 tutorial applies it, with allele
# frequencies and SDs from the training people only
train_rows <- if (!is.null(opt$train_ids)) {
  which(obj.bigSNP$fam$sample.ID %in% fread(opt$train_ids, header = FALSE)$V2)
} else rows_along(genotype)
af_val <- big_colstats(genotype, ind.row = train_rows, ind.col = info_snp$`_NUM_ID_`,
                       ncores = NCORES)$sum / (2 * length(train_rows))
sd_val <- sqrt(2 * af_val * (1 - af_val))
sd_ss  <- with(info_snp, 2 / sqrt(n_eff * beta_se^2 + beta^2))
is_bad <- sd_ss < (0.5 * sd_val) | sd_ss > (sd_val + 0.1) | sd_ss < 0.05 | sd_val < 0.05 |
          abs(af_val - info_snp$freq) > 0.1 | pmin(info_snp$freq, 1 - info_snp$freq) < 0.01
is_bad[is.na(is_bad)] <- TRUE
cat("QC removed", sum(is_bad), "of", length(is_bad), "variants (",
    round(100 * mean(is_bad), 2), "%)\n")
info_snp <- info_snp[!is_bad, ]

corr <- NULL
ld   <- NULL

for (chr in 1:22) {
  ind.chr <- which(info_snp$chr == chr)
  if (length(ind.chr) == 0) next

  # Position of each kept variant within this chromosome's reference block
  ind.block <- match(info_snp$ldref_id[ind.chr], which(info$chr == chr))

  corr0 <- readRDS(file.path(opt$ldref_dir,
                             paste0("LD_with_blocks_chr", chr, ".rds")))[ind.block, ind.block]

  if (is.null(corr)) {
    ld   <- Matrix::colSums(corr0^2)
    corr <- as_SFBM(corr0, tmp, compact = TRUE)
  } else {
    ld   <- c(ld, Matrix::colSums(corr0^2))
    corr$add_columns(corr0, nrow(corr))
  }
}

# ── LD score regression ───────────────────────────────────────────────────────

fam.order <- as.data.table(obj.bigSNP$fam)
setnames(fam.order, c("family.ID", "sample.ID"), c("FID", "IID"))

df_beta <- info_snp[, c("beta", "beta_se", "n_eff", "_NUM_ID_")]

ldsc <- snp_ldsc(
  ld,
  length(ld),
  chi2        = (df_beta$beta / df_beta$beta_se)^2,
  sample_size = df_beta$n_eff,
  blocks      = NULL
)

cat("LDSC h2 =", round(ldsc[["h2"]], 4), "\n")
if (ldsc[["h2"]] < 1e-4) cat("LDSC h2 below 1e-4, floored\n")
h2_est <- max(ldsc[["h2"]], 1e-4)
h2_floored <- ldsc[["h2"]] < 1e-4

# The SE of h2, from a second regression with jackknife blocks; the estimate the model
# uses is the one above, unchanged
h2_se <- tryCatch(
  snp_ldsc(ld, length(ld), chi2 = (df_beta$beta / df_beta$beta_se)^2,
           sample_size = df_beta$n_eff, blocks = 200, ncores = NCORES)[["h2_se"]],
  error = function(e) { cat("h2 SE not computed:", conditionMessage(e), "\n"); NA_real_ })
cat("LDSC h2 SE =", round(h2_se, 4), "\n")

# The fold's diagnostics, written now so a crash still leaves a record, and again at the end
diag_file <- paste0(sub("\\.[^.]*$", "", opt$out), "_diagnostics.csv")
write_diag <- function(chains_kept, completed) {
  fwrite(data.table(h2 = ldsc[["h2"]], h2_se = h2_se, h2_floored = h2_floored,
                    chains_kept = chains_kept, completed = completed), diag_file)
}
write_diag(NA_integer_, FALSE)

# ── Null model ────────────────────────────────────────────────────────────────

y <- pheno[fam.order, on = c("FID", "IID")]

# The diagnostic R-squared is fitted on the training people only; the test fold is scored, never fitted
reg_rows <- if (!is.null(opt$train_ids)) {
  which(fam.order$IID %in% fread(opt$train_ids, header = FALSE)$V2)
} else seq_len(nrow(fam.order))
y_reg <- y[reg_rows]

# Build the formula from whatever covariates came in, rather than assuming
# Sex and exactly 6 PCs. AGOG supplies 7 PCs and age, and no sex column.
cov.names <- setdiff(names(y), c("FID", "IID", "phenotype", "PRS"))
cov.names <- cov.names[sapply(y[, ..cov.names], function(x) length(unique(na.omit(x))) > 1)]

null.formula <- as.formula(
  paste("phenotype ~", paste(cov.names, collapse = "+")))

if (opt$trait == "bin") {
  if (!requireNamespace("fmsb", quietly = TRUE)) {
    install.packages("fmsb", repos = "http://cran.us.r-project.org")
  }
  library(fmsb)

  null.model <- glm(null.formula, data = y_reg, family = binomial)
  null.r2    <- fmsb::NagelkerkeR2(null.model)$R2

} else if (opt$trait == "quant") {
  null.model <- lm(null.formula, data = y_reg)
  null.r2    <- summary(null.model)$r.squared

} else {
  stop("Trait type not recognised. Please specify either 'bin' or 'quant'.")
}

# ── Helper: extract R² appropriate to trait type ──────────────────────────────

get_r2 <- function(formula, data, trait) {
  if (trait == "bin") {
    fit <- glm(formula, data = data, family = binomial)
    return(fmsb::NagelkerkeR2(fit)$R2)
  } else {
    fit <- lm(formula, data = data)
    return(summary(fit)$r.squared)
  }
}

# ── PRS regression formula ────────────────────────────────────────────────────

reg.formula <- as.formula(
  paste("phenotype ~ PRS +", paste(cov.names, collapse = "+")))

# ── Fit PRS model ─────────────────────────────────────────────────────────────

prs_output  <- NULL
beta_output <- NULL

if (opt$model == "inf") {

  beta_inf <- snp_ldpred2_inf(corr, df_beta, h2 = h2_est)

  pred_inf <- big_prodVec(
    genotype, beta_inf,
    ind.row = 1:nrow(genotype),
    ind.col = info_snp$`_NUM_ID_`
  )

  reg.dat     <- y_reg
  reg.dat$PRS <- pred_inf[reg_rows]
  full.r2     <- get_r2(reg.formula, reg.dat, opt$trait)

  result <- data.table(infinitesimal = full.r2 - null.r2, null = null.r2)

  prs_output  <- data.table(FID = fam.order$FID, IID = fam.order$IID, PRS = pred_inf)
  beta_output <- data.table(rsid = info_snp$rsid, chr = info_snp$chr, pos = info_snp$pos,
                             a1 = info_snp$a1, a0 = info_snp$a0, beta = beta_inf)

} else if (opt$model == "grid") {

  p_seq  <- signif(seq_log(1e-4, 1, length.out = 17), 2)
  h2_seq <- round(h2_est * c(0.7, 1, 1.4), 4)

  grid.param <- expand.grid(p = p_seq, h2 = h2_seq, sparse = c(FALSE, TRUE))
  beta_grid  <- snp_ldpred2_grid(corr, df_beta, grid.param, ncores = NCORES)

  pred_grid <- big_prodMat(
    genotype, beta_grid,
    ind.col = info_snp$`_NUM_ID_`
  )

  reg.dat <- y_reg
  max.r2  <- 0
  best.i  <- 1

  for (i in 1:ncol(pred_grid)) {
    reg.dat$PRS <- pred_grid[reg_rows, i]
    r2.i <- get_r2(reg.formula, reg.dat, opt$trait)
    if (r2.i > max.r2) {
      max.r2 <- r2.i
      best.i <- i
    }
  }

  result <- data.table(grid = max.r2 - null.r2, null = null.r2)

  prs_output  <- data.table(FID = fam.order$FID, IID = fam.order$IID, PRS = pred_grid[, best.i])
  beta_output <- data.table(rsid = info_snp$rsid, chr = info_snp$chr, pos = info_snp$pos,
                             a1 = info_snp$a1, a0 = info_snp$a0, beta = beta_grid[, best.i])

} else if (opt$model == "auto") {

  # The LDpred2 authors' recommended settings for robustness: 30 chains with
  # p starting between 1e-4 and 0.2, no sign jumps, slightly shrunk LD, and
  # no MLE step, which matters at a few thousand people
  multi_auto <- snp_ldpred2_auto(
    corr, df_beta,
    h2_init         = h2_est,
    vec_p_init      = seq_log(1e-4, 0.2, length.out = 30),
    allow_jump_sign = FALSE,
    shrink_corr     = 0.95,
    use_MLE         = FALSE,
    ncores          = NCORES
  )

  beta_auto <- sapply(multi_auto, function(auto) auto$beta_est)

  # Keep the chains the tutorial keeps: those whose range of correlation
  # estimates is above 95% of the 95th percentile across chains. A diverged
  # chain gives NA and is dropped.
  range_est <- sapply(multi_auto, function(auto) diff(range(auto$corr_est)))
  usable    <- is.finite(range_est)
  if (!any(usable)) {
    stop("Every LDpred2-auto chain diverged, so there is nothing to average",
         call. = FALSE)
  }
  keep <- usable & range_est > (0.95 * quantile(range_est[usable], 0.95))
  if (!any(keep)) {
    cat("No chain passed the range filter, averaging all", sum(usable),
        "usable chains instead\n")
    keep <- usable
  }
  cat("Averaging", sum(keep), "of", length(range_est), "chains\n")
  chains_kept <- sum(keep)

  # Log every chain's estimates, so how many converged and where is on record
  tryCatch({
    chains <- data.table(
      chain     = seq_along(multi_auto),
      p_init    = sapply(multi_auto, function(auto) auto$p_init),
      p_est     = sapply(multi_auto, function(auto) auto$p_est),
      h2_est    = sapply(multi_auto, function(auto) auto$h2_est),
      corr_range = range_est,
      kept      = keep
    )
    print(chains, nrows = 40)
    cat("Kept chains: p_est", round(range(chains$p_est[keep]), 5),
        " h2_est", round(range(chains$h2_est[keep]), 4), "\n")
    fwrite(chains, paste0(sub("\\.[^.]*$", "", opt$out), "_chains.tsv"), sep = "\t")
  }, error = function(e) cat("Chain log skipped:", conditionMessage(e), "\n"))

  final_beta_auto <- rowMeans(beta_auto[, keep, drop = FALSE])

  pred_auto <- big_prodVec(
    genotype, final_beta_auto,
    ind.row = 1:nrow(genotype),
    ind.col = info_snp$`_NUM_ID_`
  )

  reg.dat     <- y_reg
  reg.dat$PRS <- pred_auto[reg_rows]
  full.r2     <- get_r2(reg.formula, reg.dat, opt$trait)

  result <- data.table(auto = full.r2 - null.r2, null = null.r2)

  prs_output  <- data.table(FID = fam.order$FID, IID = fam.order$IID, PRS = pred_auto)
  beta_output <- data.table(rsid = info_snp$rsid, chr = info_snp$chr, pos = info_snp$pos,
                             a1 = info_snp$a1, a0 = info_snp$a0, beta = final_beta_auto)

} else {
  stop("Model type not recognised. Please specify either 'inf', 'grid', or 'auto'.")
}

# ── Save output ───────────────────────────────────────────────────────────────

if (is.null(opt$out)) stop("Output file name must be specified.")

out_base <- sub("\\.[^.]*$", "", opt$out)

fwrite(result, opt$out, sep = "\t")
cat("Main results saved to:          ", opt$out, "\n")

if (!is.null(prs_output)) {
  # Flag the training people. LDpred2-auto never fits on them, but they are scored
  # so the age/sex-adjusted model can be fitted on the training fold. evaluate.py
  # drops them before the AUC.
  if (!is.null(opt$train_ids)) {
    train.ids <- fread(opt$train_ids, header = FALSE)$V2
    prs_output[, is_train := IID %in% train.ids]
  }

  prs_file <- paste0(out_base, "_PRS.csv")
  fwrite(prs_output, prs_file, sep = ",")
  cat("PRS scores saved to:            ", prs_file, "\n")
}

if (!is.null(beta_output)) {
  beta_file <- paste0(out_base, "_betas.csv")
  fwrite(beta_output, beta_file, sep = ",")
  cat("Posterior effect sizes saved to:", beta_file, "\n")
}

# Everything is on disk: record the run as complete, with the chains the average used
if (opt$model == "auto") write_diag(chains_kept, TRUE)
cat("Diagnostics saved to:           ", diag_file, "\n")

# Adapted from the PRS pipeline of Max Schuran, https://github.com/xamwise/nextflow_pipeline_prs (commit 9a7bc59),
# released under the MIT License; see LICENSE_nextflow_pipeline_prs in this folder.
#install.packages(c("devtools","RcppArmadillo", "data.table", "Matrix"), dependencies=TRUE)
# install_github("tshmak/lassosum")

library(lassosum)
# Prefer to work with data.table as it speeds up file reading
library(data.table)
library(methods)
library(magrittr)
# For multi-threading, you can use the parallel package and 
# invoke cl which is then passed to lassosum.pipeline
library(parallel)
library(optparse)

option_list = list(
  make_option(c("-b", "--bed"), type="character", default=NULL, 
              help="bedfile name without extension", metavar="character"),
  make_option(c("-p", "--pheno"), type="character", default=NULL, 
              help="phenotype file name", metavar="character"),
  make_option(c("-c", "--cov"), type="character", default=NULL,
              help="covariate name", metavar="character"),
  make_option(c("-z", "--pcs"), type="character", default=NULL,
              help="principal component analysis file name", metavar="character"), 
  # make_option(c("-", "--ld"), type="character", default=NULL, 
  #             help="ld file name", metavar="character"),
  make_option(c("--train_ids"), type="character", default=NULL,
              help="file naming the people to tune on (FID IID, no header)", metavar="character"),
  make_option(c("-f", "--sum_stats"), type="character", default=NULL,
              help="additional data file name", metavar="character"),
    make_option(c("-o", "--out"), type="character", default=NULL, 
              help="results data file name", metavar="character")
)  

opt_parser = OptionParser(option_list=option_list)
opt = parse_args(opt_parser)

if (is.null(opt$bed)){
  print_help(opt_parser)
  stop("At least one argument must be supplied (input bed).n", call.=FALSE)
}


# This will invoke 2 threads. 
cl <- makeCluster(2)
set.seed(100)

sum.stat <- opt$sum_stats
bfile <- opt$bed
# Read in and process the covariates. AGOG holds the PCs and age in the one
# headerless file, so there is nothing to merge when no --pcs is given.
# Need as.data.frame later as lassosum doesn't handle data.table
# covariates very well
if (is.null(opt$pcs)) {
  cov <- fread(opt$cov, header = FALSE)
  setnames(cov, colnames(cov),
           c("FID", "IID", paste0("PC", 1:(ncol(cov) - 3)), "AGE"))
} else {
  covariate <- fread(opt$cov)
  pcs <- fread(opt$pcs) %>%
    setnames(., colnames(.), c("FID","IID", paste0("PC",1:(ncol(.) - 2))))
  cov <- merge(covariate, pcs)
}

# Add sex from the .fam (1 male, 2 female), the same source and coding as the
# per-fold GWAS, without re-sorting the rows
sex <- fread(paste0(opt$bed, ".fam"), header = FALSE, select = c(2, 5),
             col.names = c("IID", "SEX"))
# The covariate file is shared across datasets, so keep only this dataset's people
cov <- cov[as.character(IID) %in% as.character(sex$IID)]
cov[, SEX := sex$SEX[match(as.character(IID), as.character(sex$IID))]]
if (!all(cov$SEX %in% c(1, 2))) {
  stop("lassosum: ", sum(!cov$SEX %in% c(1, 2)), " people have sex outside 1/2 in the .fam")
}
cat("Covariates for", nrow(cov), "people\n")
cat("Tuning covariates:", paste(setdiff(names(cov), c("FID", "IID")), collapse = " "), "\n")

# We will need the EUR.hg19 file provided by lassosum 
# which are LD regions defined in Berisa and Pickrell (2015) for the European population and the hg19 genome.
ld.file <- "EUR.hg19"
# output prefix
prefix <- opt$out
# Read in the target phenotype. AGOG keeps it in the .fam, so fall back to
# column 6 there (1 control, 2 case, -9 missing) when no --pheno is given.
if (is.null(opt$pheno)) {
  target.pheno <- fread(paste0(bfile, ".fam"), header = FALSE,
                        select = c(1, 2, 6),
                        col.names = c("FID", "IID", "phenotype"))
  target.pheno[!phenotype %in% c(1, 2), phenotype := NA]
  target.pheno[, phenotype := phenotype - 1]
} else {
  target.pheno <- fread(opt$pheno)[,c("FID", "IID", "phenotype")]
}
# Read in the summary statistics
ss <- fread(sum.stat)
# Remove P-value = 0, which causes problem in the transformation
ss <- ss[!P == 0]
# Transform the P-values into correlation
cor <- p2cor(p = ss$P,
             n = ss$N,
             sign = log(ss$OR)
)
fam <- fread(paste0(bfile, ".fam"))
fam[,ID:=do.call(paste, c(.SD, sep=":")),.SDcols=c(1:2)]

# Read the training people first: they supply the LD reference and the tuning
if (!is.null(opt$train_ids)) {
  train.ids <- fread(opt$train_ids, header = FALSE)$V2
  keep.train <- as.data.frame(target.pheno[IID %in% train.ids, .(FID, IID)])
  cat("LD reference: the", nrow(keep.train), "training people\n")
} else {
  keep.train <- NULL
}

# Run the lassosum pipeline
# The cluster parameter is used for multi-threading
# You can ignore that if you do not wish to perform multi-threaded processing
# LD is estimated from the training people only (keep.ref); everyone is scored.
out <- lassosum.pipeline(
  cor = cor,
  chr = ss$CHR,
  pos = ss$BP,
  A1 = ss$A1,
  A2 = ss$A2,
  ref.bfile = bfile,
  test.bfile = bfile,
  keep.ref = keep.train,
  LDblocks = ld.file, 
  cluster=cl,
  max.ref.bfile.n = 40000
)
# Choose lambda and s by pseudovalidation (Mak et al. 2017): the summary statistics
# against the LD of the training people's genotypes, with no phenotype anywhere.
# `keep` restricts the genotypes used without changing which people get scored.
if (!is.null(keep.train)) {
  cat("Pseudovalidation on the genotypes of", nrow(keep.train), "people from", opt$train_ids, "\n")
  target.res <- pseudovalidate(out, test.bfile = bfile, keep = keep.train,
                               plot = FALSE, cluster = cl)
} else {
  cat("No --train_ids given, pseudovalidation on everyone in the target\n")
  target.res <- pseudovalidate(out, plot = FALSE, cluster = cl)
}
# The pseudovalidation statistic of the chosen model
r2 <- max(target.res$validation.table$value)

# Score everyone with the chosen model, not just the people it was tuned on.
# out$pgs is a list over s, each a matrix of people by lambda. Match on nearest
# rather than equality, since these are floats.
if (is.null(out$pgs)) {
  stop("lassosum.pipeline returned no pgs, so the non-tuning people cannot be scored",
       call. = FALSE)
}
best.s      <- which.min(abs(out$s - target.res$best.s))
best.lambda <- which.min(abs(out$lambda - target.res$best.lambda))
cat("Best model: s =", out$s[best.s], ", lambda =", out$lambda[best.lambda], "\n")
best.prs    <- out$pgs[[best.s]][, best.lambda]

# Create PRS output dataframe
prs_output <- data.table(
  FID = target.pheno$FID,
  IID = target.pheno$IID,
  PRS = best.prs
)

# Flag the tuning people so evaluate.py drops them and scores the test fold only
if (!is.null(opt$train_ids)) {
  prs_output[, is_train := IID %in% train.ids]
}

# Extract beta coefficients for the best model using target.res$best.beta
best.beta <- target.res$best.beta

# Get SNP information from the bim file
snp.info <- fread(paste0(bfile, ".bim"))

# Check if out$test.extract exists to get the SNPs used
if (!is.null(out$test.extract) && length(out$test.extract) > 0) {
  # Extract only the SNPs that were used in the analysis
  used.snps <- snp.info[out$test.extract, ]
} else {
  # Assume all SNPs were used if test.extract is not available
  used.snps <- snp.info
}

# Make sure the number of betas matches the number of SNPs
if (length(best.beta) != nrow(used.snps)) {
  cat("Warning: Number of betas (", length(best.beta), ") doesn't match number of SNPs (", 
      nrow(used.snps), ")\n")
  # Adjust to the smaller number
  min.length <- min(length(best.beta), nrow(used.snps))
  used.snps <- used.snps[1:min.length, ]
  best.beta <- best.beta[1:min.length]
}

# Create beta output dataframe
beta_output <- data.table(
  rsid = used.snps$V2,
  chr = used.snps$V1,
  pos = used.snps$V4,
  a1 = used.snps$V5,
  a0 = used.snps$V6,
  beta = best.beta
)

# Write out the results
write.table(target.res$validation.table, 
            file=paste0(prefix, "_lassosum.txt"), 
            sep="\t", 
            row.names=F, 
            quote=F)

# Write maximum R2
write.table(r2, 
            file=paste0(prefix, "_lassosum_r2.txt"), 
            sep="\t", 
            row.names=F, 
            quote=F)

# Save PRS scores for each individual
prs_file <- paste0(prefix, "_PRS.csv")
fwrite(prs_output, prs_file, sep=",")
cat("PRS scores saved to:", prs_file, "\n")

# Save posterior effect sizes for each SNP
beta_file <- paste0(prefix, "_betas.csv")
fwrite(beta_output, beta_file, sep=",")
cat("Posterior effect sizes saved to:", beta_file, "\n")

cat("\n=== LASSOSUM RESULTS SUMMARY ===\n")
cat("Output files created:\n")
cat("  Validation results:", paste0(prefix, "_lassosum.txt"), "\n")
cat("  R-squared:", paste0(prefix, "_lassosum_r2.txt"), "\n")
cat("  PRS scores:", prs_file, "\n")
cat("  Beta coefficients:", beta_file, "\n")
cat("  Number of individuals with PRS:", nrow(prs_output), "\n")
cat("  Number of SNPs with betas:", nrow(beta_output), "\n")

# Close the cluster
stopCluster(cl)
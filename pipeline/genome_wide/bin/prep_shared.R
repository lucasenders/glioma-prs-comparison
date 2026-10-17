#!/usr/bin/env Rscript

# Convert one cohort's genotypes into bigsnpr's format once, as <bed>_shared.bk
# and <bed>_shared.rds, and cache the genetic maps beside them.
#
# Without this every fold of every bigsnpr method re-encodes the same .bed and
# re-downloads the same genetic maps -- 15 times per method, identical each time.
# LDpred-2.R picks the shared copy up automatically when it exists.
#
# Run through prep_shared.py rather than directly.

library(bigsnpr)
library(optparse)

option_list <- list(
  make_option(c("-b", "--bed"), type = "character", default = NULL,
              help = "bed file name without extension", metavar = "character")
)

opt <- parse_args(OptionParser(option_list = option_list))

if (is.null(opt$bed)) {
  stop("--bed is required", call. = FALSE)
}

shared <- paste0(opt$bed, "_shared")

if (file.exists(paste0(shared, ".rds"))) {
  cat("Already there:", paste0(shared, ".rds"), "\n")
} else {
  cat("Converting", paste0(opt$bed, ".bed"), "\n")
  snp_readBed(paste0(opt$bed, ".bed"), backingfile = shared)
  cat("Wrote", paste0(shared, ".rds"), "\n")
}

# Pull the genetic maps down once into the folder the methods will look in
obj <- snp_attach(paste0(shared, ".rds"))
cat("Caching genetic maps in", dirname(opt$bed), "\n")
invisible(snp_asGeneticPos(obj$map$chromosome, obj$map$physical.pos,
                           dir = dirname(opt$bed)))

cat("Done:", nrow(obj$fam), "people,", nrow(obj$map), "variants\n")

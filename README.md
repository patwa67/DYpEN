# DYpREG contains Julia code for the Davis-Yin prior Regularization method presented in Waldmann and Fan (2025 to be published). Included is also R code for the INLA and BGLR packages.

This package DYpREG accommodates two distinct loss functions and a regularization function that can be chosen to correspond to LASSO, ridge regression (RR) or elastic net (EN). Notably, it incorporates pre-calculated breeding values, mimicking the single-step genomic approach without the need for extensive relationship matrix computations. 

1. For the mice data, INLA file should be used first since it creates the MiceBL.txt data file'
When installing the package, we need to use install.packages() adding the URL of the R-INLA repository. For example, to install the stable version of the package, we need to type the following instruction:
install.packages("INLA",
repos = "https://inla.r-inla-download.org/R/stable", dep = TRUE)
Then, to load the package in R, we need to type
library(INLA)
In this package, the EBVs for mice data have been included.

2. Another public pig dataset that contains 3534 individuals with high-density genotypes, phenotypes of five anonymized traits, and a pedigree derived from a single nucleus line including parents and grandparents of the genotyped animals (n = 6473) that was used for full evaluation to get estimated breeding values (EBVs) \cite{Cleveland12}. Hence, in this data were the EBVs already estimated.
(Cleveland, M. A., Hickey, J. M., & Forni, S. (2012). A common dataset for genomic analysis of livestock populations. G3: Genes| Genomes| Genetics, 2(4), 429-435.) If you need, you can access the pig data use the link here: https://github.com/angelYHF/Pig-data.

3. We have given the basic settings of the code. More details about ranges of different hyperparameters for DYpEN, DYpRR, and DYpLASSO  have included in the paper.

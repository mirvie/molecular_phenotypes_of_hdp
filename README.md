# Molecular subtyping of hypertensive disorders of pregnancy

[![DOI](https://zenodo.org/badge/831186081.svg)](https://zenodo.org/badge/latestdoi/831186081)

# Overview
This repository contains code to reproduce the statistics and figures for 
**"Molecular subtyping of hypertensive disorders of pregnancy"**.

# Code Files
Each code files generates results as described: 
1. `Stats.Rmd`: R markdown for generating statistics in manuscript
2. `Figure_2abc.R`: R script for creating Figures 2a-2c
3. `Figures_2d_2e_3_4a_4b_4d_Extended_Data_1_3_Suppl_Tbl_1_NICU.ipynb`: Jupyter notebook for creating Figures 2d, 2e, 3,
4a, 4b, 4d, and Extended Data Figures 1 & 3, Supplementary Table 1, and analysis for days spent in the NICU
4. `Figure_4c.R`: R script for creating Figure 4c
5. `Figure_5a.ipynb`: Jupyter notebook for creating Figure 5a
6. `Figure_5b.R`: R script for creating Figure 5b
7. `Figure_Extended_Data_2ab_4_5.R`: R script for Extended Data Figure 2a and 2b, 4, and 5
8. `SI_proximity_to_delivery.R`: R script for "Evaluation of Collection Proximity to Delivery versus HDP Severity" in supplemental information.


# Dependencies
Dependencies are listed per code file.

## `Stats.Rmd`
* R 4.2.2
* tidyverse 2.0.0
* knitr 1.45
* arrow 10.0.1
* jsonlist 1.8.7
* correlation 0.8.4
* psychometric 2.4
* effectsize 0.8.8

## `Figure_2abc.R`, `Figure_4c.R`, `Figure_5b.R`
* Requirements listed for [Stats.Rmd](#StatsRmd)
* patchwork 1.2.0
* roxygen2 7.3.1
* colorspace 2.0-3
* ggtext 0.1.2

## `Figures_2d_2e_3_4a_4b_4d_Extended_Data_1_3_Suppl_Tbl_1_NICU.ipynb`
* python 3.12.4
* numpy 1.26.4
* pandas 2.2.0
* matplotlib_venn 1.1.1
* scipy 1.14.0
* matplotlib 3.8.4
* seaborn 0.13.2
* sklearn 1.5.1
* statsmodels 0.14.2
* Figures_2d_2e_3_4a_4b_4d_Extended_Data_1_3_Suppl_Tbl_1_NICU_helper.py (dependencies outlined above)

## `Figure_5a.ipynb`
* python 3.10.13
* jupyter 1.0.0
* numpy 1.26.4
* pandas 1.5.3
* matplotlib 3.8.3
* seaborn 0.13.2
* sklearn 1.4.1.post1 
* fig5a_helper.py (dependencies outlined above)

## `Figure_Extended_Data_2ab_4_5.R`
* R 4.4.1
* cogena 1.21.2
* fgsea 1.30.0
* ggplot2 3.5.1
* msigdbr 7.5.1
* tidyverse 2.0.0
* ggpubr 0.6.0

## `SI_proximity_to_delivery.R`
* R 4.2.2
* tidyverse 2.0.0
* arrow 10.0.1
* Hmisc 5.1-3

# Installation Guide

## Installing dependencies for R and R markdown

### `Figure_2abc.R`, `Figure_4c.R`, `Figure_5b.R`, `SI_proximity_to_delivery.R`, and `Stats.Rmd`
To install dependencies, perform the following:
```
1. Install R 4.2.2 from https://cran.r-project.org/bin/windows/base/old/
2. Install RStudio from https://posit.co/download/rstudio-desktop/
3. Install `devtools` via install.packages("devtools"). 
4. Install the listed dependencies via calls to devtools::install_version("PACKAGE", version = "VERSION", repos = "http://cran.us.r-project.org"), where PACKAGE is the package name and VERSION is the package version
```

### `Figure_Extended_Data_2ab_4_5.R`
To install packages, run
```
# CRAN packages
list_of_cran_packages <- c("fgsea","ggplot2", "msigdbr", "tidyverse", "ggpubr")
new.packages <- setdiff(list_of_cran_packages, installed.packages()[,"Package"])
if(length(new.packages)) install.packages(new.packages)

#Devtools packages
devtools::install_github("zhilongjia/cogena")

#Bioconductor packages
if (!require("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
BiocManager::install("Biobase", force = TRUE)

# attach libraries if necessary/desired
# read_library <- function(...) {
#     obj <- eval(substitute(alist(...)))
#     #print(obj)
#     return(invisible(lapply(obj, \(x) library(toString(x), character.only=TRUE))))
# }
#
# read_library(cogena, fgseq, ggplot2, msigdbr, tidyverse)
```

## Installing dependencies for Jupyter notebooks
To run the Jupyter notebook, ensure minimal dependencies are installed:
* python (see versions above)
* conda (see versions above)
* jupyter (see versions above)

Then install the library requirements listed above with `conda install`.

# Inputs
The code is setup to run on the manuscript data. Manuscript data are available with a 
signed data use agreement to protect identifiable data. Please contact 
`research@mirvie.com`.

To run the code on your own data, the file structure must match as described below and
column labels should be edited as appropriate in the scripts.

## Input files required by most scripts:
Most scripts (excluding 
`Figures_2d_2e_3_4a_4b_4d_Extended_Data_1_3_Suppl_Tbl_1_NICU.ipynb` and 
`Figure_5a.ipynb`) require `sample_data.feather`, which is a feather formatted 
sample-wise dataframe containing samples as rows and gene names and metadata 
information as columns. The gene expression values here are corrected log2cpm.

Input file for `Figures_2d_2e_3_4a_4b_4d_Extended_Data_1_3_Suppl_Tbl_1_NICU.ipynb`
is `sample_data_scaled.feather`, which is a feather formatted 
sample-wise dataframe containing samples as rows and gene names and metadata 
information as columns. The gene expression values here are scaled and corrected
log2cpm.

Input files for `Figure_5a.ipynb` are samples from clinical validation that are not 
accessible to the modeling team. Code can be run with alternative input files for 
testing.

## Input files additionally required by `Stats.Rmd`
`Stats.Rmd` additionally requires `genes_space.json`, a JSON containing gene names 
comprising the search space. 

## Input files additionally required by `Figures_2d_2e_3_4a_4b_4d_Extended_Data_1_3_Suppl_Tbl_1_NICU.ipynb`
In addition to the `Figures_2d_2e_3_4a_4b_4d_Extended_Data_1_3_Suppl_Tbl_1_NICU_helper.py` 
script, this notebook requires 3 additional input data files:
* `genes_space.json`: a JSON containing gene names comprising the search space. 

The following 3 input files are feather formatted gene-wise dataframes containing 
genes as rows and p-values, effect sizes, and analysis labels as columns to generate 
the following figures:
* `fig3_de_data.feather`:  input for generating Figure 3
* `fig4ab_de_data.feather`: input for generating Figures 4a, 4b
* `suppl_tbl1_data.feather`: input for generating Supplementary Table 1

## Input files additionally required by `Figure_Extended_Data_1abc.R`
This script additionally requires a GMT annotation file.

# Running the Code
Note that input files must be in the same directory as the script/notebook that is run.

## Running R and R markdown

Expected run times should complete in less than a minute.

### Instructions for running `Stats.Rmd`
```
1. Install needed dependencies as listed above.
2. Open the file in Rstudio.
3. Hit the "knit" button in Rstudio.
```

### Instructions for running `Figure_2abc.R`, `Figure_4c.R`, `Figure_5b.R`, and `SI_proximity_to_delivery.R`
```
1. Install needed dependencies as listed above.
2. Open the file in Rstudio.
3. Execute code in order. Can execute the entire file from the R console via source("SCRIPTNAME.R", echo = TRUE).
```

### Instructions for running `Figure_extended_data_2AB_4_5.R`
The script can be sourced  from R console or in Rstudio environment. 

## Running Jupyter Notebooks

### Instructions for running `Figures_2d_2e_3_4a_4b_4d_Extended_Data_1_3_Suppl_Tbl_1_NICU.ipynb` or `Figure_5a.ipynb`:
```
1. Start the Jupyter server in the folder containing the notebook.
2. From the Jupyter browser, click on the *.ipynb to open it.
3. In the notebook, change Kernel to the conda environment created with the required dependenices installed.
4. Click on "Kernel" > "Restart & Run All"
```

Expected Run Times
* `Figures_2d_2e_3_4a_4b_4d_Extended_Data_1_3_Suppl_Tbl_1_NICU.ipynb`: Expected run times should complete in less 
    than a minute.
* `Figure_5a.ipynb`: Expected run time is less than 1 hour.

# Software License
[Creative Commons license CC BY-NC v4.0](LICENSE.txt)

# Prospectively validated predictor reveals first molecular phenotypes of hypertensive disorders of pregnancy 

[![DOI](https://zenodo.org/badge/831186081.svg)](https://zenodo.org/doi/10.5281/zenodo.12786388)

# Overview
This repository contains code to reproduce the statistics and figures for 
**"Prospectively validated predictor reveals first molecular phenotypes of
hypertensive disorders of pregnancy"**.

# Code Files
Each code files generates results as described: 
1. `Stats.Rmd`: R markdown for generating statistics in manuscript
2. `Figure_2abc.R`: R script for creating Figures 2a-2c
3. `Figure_2c_3_4ab_Extended_Data_2.ipynb`: Jupyter notebook for creating Figures 2c, 3,
4a, 4b, and Extended Data Figure 2
4. `Figure_4c.R`: R script for creating Figure 4c
5. `Figure_5a.ipynb`: Jupyter notebook for creating Figure 5a
6. `Figure_5b.R`: R script for creating Figure 5b
7. `Figure_Extended_Data_1ab.R`: R script for Extended Data Figure 1a and 1b
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

## `Figure_2ab.R`, `Figure_4c.R`, `Figure_5b.R`
* Requirements listed for [Stats.Rmd](#StatsRmd)
* patchwork 1.2.0
* roxygen2 7.3.1
* colorspace 2.0-3
* ggtext 0.1.2

## `Figure_2c_3_4ab_Extended_Data_2.ipynb`
* python 3.12.4
* pandas 2.2.2
* numpy 2.0.0
* matplotlib 3.8.4
* seaborn 0.13.2
* scipy 1.14.0
* sklearn 1.5.1
* helper_functions.py (dependencies outlined above)

## `Figure_5a.ipynb`
* python 3.10.13
* jupyter 1.0.0
* numpy 1.26.4
* pandas 1.5.3
* matplotlib 3.8.3
* seaborn 0.13.2
* sklearn 1.4.1.post1 
* fig5a_helper.py (dependencies outlined above)

## `Figure_Extended_Data_1ab.R`
* R 4.4.1
* cogena 1.21.2
* fgsea 1.30.0
* ggplot2 3.5.1
* msigdbr 7.5.1
* tidyverse 2.0.0

# Installation Guide

## Installing dependencies for R and R markdown

### `Figure_2ab.R`, `Figure_4c.R`, `Figure_5b.R`, and `Stats.Rmd`
To install dependencies, perform the following:
```
1. Install R 4.2.2 from https://cran.r-project.org/bin/windows/base/old/
2. Install RStudio from https://posit.co/download/rstudio-desktop/
3. Install `devtools` via install.packages("devtools"). 
4. Install the listed dependencies via calls to devtools::install_version("PACKAGE", version = "VERSION", repos = "http://cran.us.r-project.org"), where PACKAGE is the package name and VERSION is the package version
```

### `Figure_Extended_Data_1ab.R`
To install packages, run
```
# CRAN packages
list_of_cran_packages <- c("fgsea","ggplot2", "msigdbr", "tidyverse")
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
Most scripts (excluding `Figure_5a.ipynb`) require `sample_data.feather`, which is a 
feather formatted sample-wise dataframe containing samples as rows and gene names and 
metadata information as columns.

Input files for `Figure_5a.ipynb` are samples from clinical validation that are not 
accessible to the modeling team. Code can be run with alternative input files for 
testing.

## Input files additionally required by `Stats.Rmd`
`Stats.Rmd` additionally requires `genes_space.json`, a JSON containing gene names 
comprising the search space. 

## Input files additionally required by `Figure_2c_3_4ab_Extended_Data_2.ipynb`
In addition to the `helper_functions.py` script, this notebook requires 4 additional
input data files:
* `genes_space.json`: a JSON containing gene names comprising the search space. 

This following input files that are feather formatted gene-wise dataframes containing 
genes as rows and p-values, effect sizes, and analysis labels as columns to generate 
the following figures:
* `fig3_de_data.feather`:  input for generating Figure 3
* `fig4ab_de_data.feather`: input for generating Figures 4a, 4b
* `ed_fig2_de_data.feather`: input for generating Figure 2

## Input files additionally required by `Figure_Extended_Data_1ab.R`
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

### Instructions for running `Figure_2abc.R`, `Figure_4c.R`, `Figure_5b.R`,
```
1. Install needed dependencies as listed above.
2. Open the file in Rstudio.
3. Execute code in order. Can execute the entire file from the R console via source("SCRIPTNAME.R", echo = TRUE).
```

### Instructions for running `Figure_extended_data_1AB.R`
The script can be sourced  from R console or in Rstudio environment. 

## Running Jupyter Notebooks

Instructions for running `Figure_2c_3_4ab_Extended_Data_2.ipynb` or `Figure_5a.ipynb`:
```
1. Start the Jupyter server in the folder containing the notebook.
2. From the Jupyter browser, click on the *.ipynb to open it.
3. In the notebook, change Kernel to the conda environment created with the required dependenices installed.
4. Click on "Kernel" > "Restart & Run All"
```

Expected Run Times
* `Figure_2c_3_4ab_Extended_Data_2.ipynb`: Expected run times should complete in less 
    than a minute.
* `Figure_5a.ipynb`: Expected run time is less than 1 hour.

# Software License
[Creative Commons license CC BY-NC v4.0](LICENSE.txt)

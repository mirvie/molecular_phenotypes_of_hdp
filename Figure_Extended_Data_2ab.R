# This scripts generates figure panels for extended data Figure 1AB, 4, and 5 and supplementary
# figures 1 and 2, given full dataset and gmt annotation.
######################################### Prerequisites 
##################### Package installation
#CRAN packages
install.packages("httr")

list_of_cran_packages <- c("fgsea","ggplot2", "msigdbr", "tidyverse", "ggpubr", "svglite")

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
#     return(invisible(lapply(obj, function(x)library(toString(x), character.only=TRUE))))
# }
# 
# read_library(cogena, fgseq, ggplot2, tidyverse)


############################### Gene lists
# The specified gmt files can be downloaded from 
#https://www.gsea-msigdb.org/gsea/msigdb/collections.jsp via "Gene Symbols" link
#using msigDB API following email registration

#msigdb CGP: chemical and genetic perturbations
msig_C2_CGP <- cogena::gmt2list("./inputs/c2.cgp.v2023.2.Hs.symbols.gmt") 
#msigdb CP: Canonical pathways
msig_C2_CP <- cogena::gmt2list("./inputs/c2.cp.v2023.2.Hs.symbols.gmt")
#msigdb C5: ontology gene sets
msig_C5_GO <- cogena::gmt2list("./inputs/c5.all.v2023.2.Hs.symbols.gmt")
#msigdb Cell type signature gene list
msig_C8 <- cogena::gmt2list("./inputs/c8.all.v2023.2.Hs.symbols.gmt")
#stringDB PPI
ppi_list <- readRDS("./inputs/PPI_list.rds")

#Alternatively, a slightly different gene list can constructed using msigdbr package as follows:
msig_C5_GO_df <- msigdbr::msigdbr(species = "Homo sapiens", category = "C5")
msig_C5_GO <- split(msig_C5_GO_df$gene_symbol,msig_C5_GO_df$gs_name)

####################################### data
sample_data <- arrow::read_feather("/mnt/marge/storage/users/elaine/analysis/pe/2024-0718_Data_generation_for_manuscript_figure_code_share/sample_data.feather")
nmrf_ids <- sample_data %>% filter(is_mrf == TRUE & is_green_triangle == TRUE) %>% pull(sample_index)
pe_ids <- sample_data %>% filter(is_pe == TRUE) %>% pull(sample_index)
other_pe_ids <- setdiff(pe_ids, nmrf_ids)

sample_genes <- colnames(sample_data[10:17])

####################################### set contrast
#' Contrast between two groups of samples for fgsea input
#'
#' @param df - data frame with one column containing _ids and other columns containing counts for genes in full_gene_list
#' @param comparator1_ids - a vector of sample_index corresponding to ids for comparator 1
#' @param comparator2_ids - a vector of sample_index corresponding to ids for comparator 2
#' @param full_gene_list - a character vector of gene names found in the df
#'
#' @return named numerical vector of the length of full_gene_list
#' @export NA
#'
#' @examples nmrf_other_pe_d <- get_ranked_meandiff_for_fgsea(
#' df = sample_data, 
#' comparator1_ids = nmrf_ids, 
#' comparator2_ids = other_pe_ids, 
#' full_gene_list = sample_genes)
get_ranked_meandiff_for_fgsea <- function(df, comparator1_ids, comparator2_ids, full_gene_list){
  cmpr1 <- apply(df %>% filter(sample_index %in% comparator1_ids) %>% select(all_of(full_gene_list)), 2, mean)
  cmpr2 <- apply(df %>% filter(sample_index %in% comparator2_ids) %>% select(all_of(full_gene_list)), 2, mean)
  d <- sort(cmpr1 - cmpr2)
  return(d)
}

# sample_nmrf_other_pe_d <- get_ranked_meandiff_for_fgsea(
# df = sample_data, 
# comparator1_ids = nmrf_ids, 
# comparator2_ids = other_pe_ids, 
# full_gene_list = sample_genes)

####################################### running gsea
fgseaRes_sample_nmrf_other_pe_d <- fgsea::fgsea(pathways = ppi_list, 
                                                stats    =  sample_nmrf_other_pe_d
)

####################################### generating figure

#' Title - Plots gene sets uniquely enriched (up or down) in one comparison but not in another
#'
#' @param df - fgsea output
#' @param top_n - number of top gene sets to display
#' @param name1 - name of the contrast 1 to be printed in the title
#' @param name2 - name of the contrast 2 to be printed in the title
#' @param padj_cutoff - FDR cutoff: ]0;1[]

#'
#' @return ggplot2 object
#' @export NA
#'
#' @examples Fig1_extended_data_1ab_sample <- plot_gsea_singles_new(
#df = fgseaRes_sample_nmrf_other_pe_d,
#top_n = 20, name1 = "PAPPA+", name2 = "PAPPA2-")
plot_gsea_singles_new <- function(df, top_n, padj_cutoff=NULL, name1, name2){
  
  x_breaks = c(0, 0.5, 1)
  
  if(!is_null(padj_cutoff))
{  UP_plot_input <- df %>%
    filter(padj < padj_cutoff & NES > 0) %>%
    mutate(core = lengths(leadingEdge)/size) %>% 
    mutate(abs_NES = abs(NES))}
  else
{  UP_plot_input <- df %>%
    filter(NES > 0) %>%
    mutate(core = lengths(leadingEdge)/size) %>% 
    mutate(abs_NES = abs(NES))}
  
  if(!is_null(padj_cutoff))
 { DWN_plot_input <- df %>%
    filter(padj < padj_cutoff & NES < 0) %>%
    mutate(core = lengths(leadingEdge)/size, abs_NES = abs(NES))}
  else
  { DWN_plot_input <- df %>%
    filter(NES < 0) %>%
    mutate(core = lengths(leadingEdge)/size, abs_NES = abs(NES))}
    
  UP_plot_input_byNES <-  UP_plot_input %>% 
    slice_max(abs(NES), n = top_n) 
  
  p_UP_byNES <- UP_plot_input_byNES %>%
    ggplot(., aes(x=core, y=reorder(pathway,core), fill=-log10(padj), size=abs(NES))) +geom_point(shape = 21, color = "black")  + 
    theme_bw() + 
    theme(axis.text.y = element_text(size=7), plot.title = element_text(hjust=0.7))  + 
    scale_size_continuous(name = "NES",
                          breaks = seq(round(min(UP_plot_input_byNES$abs_NES),1), round(max(UP_plot_input_byNES$abs_NES), 1), by = 0.2)
                          #,
                          # labels = function(x) {
                          #   # Custom function to add sign to legend labels
                          #   sign <- ifelse(max(UP_plot_input_byNES$NES) < 0, "-", "+")
                          #   paste0(sign, abs(x))
                          # }
    ) + 
    scale_x_continuous(limits = c(0, 1), breaks = x_breaks) +
    #xlim(0, 1) +
    coord_fixed(ratio = 0.25) +
    guides(
      fill = guide_colorbar(order = 1),
      size = guide_legend(order = 2)) + 
    labs(x="Fraction of gene set", y="Gene sets", 
         title = paste0("Gene sets UP-regulated in ", name1, " v. " ,  name2, " (top ", top_n, ")"),  size = "NES") 
  
  #color = guide_legend(order = 1),
  #color = "-log10(padj)",
  
  UP_plot_input_byPval <-  UP_plot_input %>%
    slice_min(pval, n = top_n)
  
  p_UP_byPval <- UP_plot_input_byPval %>%
    slice_min(pval, n = top_n) %>% 
    ggplot(., aes(x=core, y=reorder(pathway,core), fill=-log10(padj), size=abs(NES))) +geom_point(shape = 21, color = "black") + 
    theme_bw() + 
    theme(axis.text.y = element_text(size=7), plot.title = element_text(hjust=0.7)) + 
    scale_size_continuous(name = "NES",
                          breaks = seq(round(min(UP_plot_input_byPval$abs_NES),1), round(max(UP_plot_input_byPval$abs_NES),1), by = 0.2)
                          # labels = function(x) {
                          #   # Custom function to add sign to legend labels
                          #   sign <- ifelse(max(UP_plot_input_byPval$NES) < 0, "-", "+")
                          #   paste0(sign, abs(x))
                          # }
    ) + 
    scale_x_continuous(limits = c(0, 1), breaks = x_breaks) +
    #xlim(0, 1) +
    coord_fixed(ratio = 0.25) +
    guides(
      fill = guide_colorbar(order=1),
      size = guide_legend(order = 2)) + 
    labs(x="Fraction of gene set", y="Gene sets", 
         title = paste0("Gene sets UP-regulated in ", name1, " v. " ,  name2, " (top ", top_n, ")"),  size = "NES")
  
  DWN_plot_input_byNES <- DWN_plot_input %>%
    slice_max(abs(NES), n = top_n)
  
  p_DWN_byNES <- DWN_plot_input_byNES %>%
    slice_max(abs(NES), n = top_n) %>% 
    #slice_min(pval, n=20) %>% 
    ggplot(., aes(x=core, y=reorder(pathway,core), fill=-log10(padj), size=abs(NES))) +geom_point(shape = 21, color = "black") +  
    theme_bw() + 
    theme(axis.text.y = element_text(size=7), plot.title = element_text(hjust=0.7)) + 
    scale_size_continuous(name = "NES",
                          breaks = seq(round(min(DWN_plot_input_byNES$abs_NES),1), round(max(DWN_plot_input_byNES$abs_NES),1), by = 0.2)
                          # ,
                          # labels = function(x) {
                          #   # Custom function to add sign to legend labels
                          #   sign <- ifelse(max(DWN_plot_input_byNES$NES) < 0, "-", "+")
                          #   paste0(sign, abs(x))
                          # }
    ) + 
    scale_x_continuous(limits = c(0, 1), breaks = x_breaks) +
    #xlim(0, 1) +
    coord_fixed(ratio = 0.25) +
    guides(
      fill = guide_colorbar(order=1),
      size = guide_legend(order = 2)) + 
    labs(x="Fraction of gene set", y="Gene sets", 
         title = paste0("Gene sets UP-regulated in ", name2, " v. " ,  name1, " (top ", top_n, ")"),  size = "NES")
  
  DWN_plot_input_byPval <- DWN_plot_input %>%
    slice_min(pval, n = top_n)  
  
  p_DWN_byPval <- DWN_plot_input_byPval %>%
    #slice_max(abs(NES), n = top_n) %>% 
    slice_min(pval, n = top_n) %>% 
    ggplot(., aes(x=core, y=reorder(pathway,core), fill =-log10(padj), size=abs(NES))) +geom_point(shape = 21, color = "black") + 
    theme_bw() + 
    theme(axis.text.y = element_text(size=7), plot.title = element_text(hjust=0.7)) + 
    scale_size_continuous(name = "NES",
                          breaks = seq(round(min(DWN_plot_input_byPval$abs_NES),1), round(max(DWN_plot_input_byPval$abs_NES),1), by = 0.2)
                          # ,
                          # labels = function(x) {
                          #   # Custom function to add sign to legend labels
                          #   sign <- ifelse(max(DWN_plot_input_byPval$NES) < 0, "-", "+")
                          #   paste0(sign, abs(x))
                          # }
    ) + 
    scale_x_continuous(limits = c(0, 1), breaks = x_breaks) +
    #xlim(0, 1) +
    coord_fixed(ratio = 0.25) +
    guides(
      fill = guide_colorbar(order=1),
      size = guide_legend(order = 2)) + 
    labs(x="Fraction of gene set", y="Gene sets", 
         title = paste0("Gene sets UP-regulated in ", name2, " v. " ,  name1, " (top ", top_n, ")"),  size = "NES")
  
  
  
  res <- list(p_UP_byNES = p_UP_byNES, p_UP_byPval = p_UP_byPval, p_DWN_byNES = p_DWN_byNES, p_DWN_byPval = p_DWN_byPval)
  return(res)
  
}

# plot_gsea_singles_new(
#   df = fgseaRes_sample_nmrf_other_pe_d,
#   top_n = 20, name1 = "PAPPA+", name2 = "PAPPA2-", padj_cutoff = NULL)

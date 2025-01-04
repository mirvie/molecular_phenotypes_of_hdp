library(tidyverse)
library(ggplot2)

df <- arrow::read_feather("df_paper_samples_log2cpm_batchcorr_techcorr.feather")
df$delivery_proximity <- df$delivery_ga - df$collection_ga
df$time_to_diagnosis <- df$pe_ga - df$collection_ga
df_gt <- df[df$is_green_triangle %in% TRUE,]

# Lead time --------------------------------------------------------------------

# GT
Hmisc::describe(df[(df$is_green_triangle == TRUE), 'delivery_proximity'] * 7)
Hmisc::describe(df[(df$is_green_triangle == TRUE) & (df$is_nmrf), 'delivery_proximity'] * 7)


# SI analysis: accounting for predictions --------------------------------------

# gene_pred ~ delivery + proximity
boot_fn <- function(data, indices) {
  fit <- lm(pe_pred_logit_genes_only ~ delivery_ga + delivery_proximity, data = data[indices, ])
  return(coef(fit))
}

boot_results <- boot::boot(df_gt, boot_fn, R = 10000)
boot_coefs <- data.frame(boot_results$t)
colnames(boot_coefs) <- coef(lm(pe_pred_logit_genes_only ~ delivery_ga + delivery_proximity, data = df_gt)) %>% names()
sapply(data.frame(boot_coefs), function(x) 2 * min(mean(x < 0), 1 - mean(x < 0)) ) # pvals

# SI analysis: accounting for dga using predictions ----------------------------

# delivery ~ gene_pred + proximity
boot_fn_dga <- function(data, indices) {
  fit <- lm(delivery_ga ~ pe_pred_logit_genes_only + delivery_proximity, data = data[indices, ])
  return(coef(fit))
}

boot_results_dga <- boot::boot(df_gt, boot_fn_dga, R = 10000)
boot_coefs_dga <- data.frame(boot_results_dga$t)
colnames(boot_coefs_dga) <- coef(lm(delivery_ga ~ pe_pred_logit_genes_only + delivery_proximity, data = df_gt)) %>% names()
sapply(data.frame(boot_coefs_dga), function(x) 2 * min(mean(x < 0), 1 - mean(x < 0)) )


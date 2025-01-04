library(tidyverse)
library(patchwork)

df <- arrow::read_feather("sample_data.feather")

# Prepare data ------------------------------------------------------------------

#' Compute Cohen's d of PAPPA2 for granular HDP subgroups.
#'
#' Samples where the HDP subgroup could not be determined are removed.
#'
#' @param data a data frame containing PAPPA2 and hdp_vanity_group columns
#'
#' @return a data frame where each row is a subgroup, with columns for the effect and sample size
compute_hdp_effects <- function(data) {
  reference_group <- data[data$hdp_vanity_group == "Control", ]$PAPPA2
  effect_sizes <- data %>%
    select(hdp_vanity_group, PAPPA2) %>%
    filter(hdp_vanity_group != "PE, Ambiguous") %>%
    group_by(hdp_vanity_group) %>%
    summarise(
      effect = effectsize::cohens_d(PAPPA2, y = reference_group, adjust = FALSE),
      n = n()
    )
  effect_sizes$effect <- effect_sizes$effect$Cohens_d
  effect_sizes$hdp_vanity_group <- factor(effect_sizes$hdp_vanity_group,
    levels = c(
      "Control",
      "GHTN",
      "Postpartum PE",
      "Term PE,  Not Severe, Term Diagnosis",
      "Term PE, Not Severe, Preterm Diagnosis",
      "Term PE, Severe, Term Diagnosis",
      "Term PE, Severe, Preterm Diagnosis",
      "Preterm PE, Not Severe",
      "Preterm PE, Severe"
    )
  )

  effect_sizes <- effect_sizes %>%
    arrange(desc(hdp_vanity_group))
  effect_sizes
}

#' Compute Cohen's d of PAPPA2 for coarse HDP subgroups.
#'
#' Parses factor labels into the terminology used by the manuscript (e.g. "PAPPA2+").
#'
#' @param data a data frame containing PAPPA2 and hdp_severity columns
#'
#' @return a data frame where each row is a subgroup, with columns for the effect and sample size
compute_hdp_effects_coarse <- function(data) {
  reference_group <- data[data$hdp_severity == "control", ]$PAPPA2
  effect_sizes_coarse <- data %>%
    select(hdp_severity, PAPPA2) %>%
    group_by(hdp_severity) %>%
    summarise(
      effect = effectsize::cohens_d(PAPPA2,
        y = reference_group,
        adjust = FALSE
      ),
      n = n()
    )
  effect_sizes_coarse$effect <- effect_sizes_coarse$effect$Cohens_d
  effect_sizes_coarse[effect_sizes_coarse$hdp_severity == "other_hdp", "hdp_severity"] <- "Immune Driven"
  effect_sizes_coarse[effect_sizes_coarse$hdp_severity == "preterm_pe", "hdp_severity"] <- "Placenta Driven"
  effect_sizes_coarse[effect_sizes_coarse$hdp_severity == "control", "hdp_severity"] <- "Non HDP"
  effect_sizes_coarse$hdp_severity <- factor(effect_sizes_coarse$hdp_severity,
    levels = c(
      "Non HDP",
      "Immune Driven",
      "Placenta Driven"
    )
  )

  effect_sizes_coarse <- effect_sizes_coarse %>%
    arrange(desc(hdp_severity))
  effect_sizes_coarse
}


# fine groups
effect_sizes_all <- compute_hdp_effects(df)
effect_sizes_all$grouping <- "Molecular Phenotype"
effect_sizes_all$grouping_nmrf <- "All Pregnancies"

effect_sizes_mrf <- compute_hdp_effects(df %>% filter(is_mrf == TRUE))
effect_sizes_mrf$grouping <- "Molecular Phenotype"
effect_sizes_mrf$grouping_nmrf <- "MRF Pregnancies"

effect_sizes_nmrf <- compute_hdp_effects(df %>% filter(is_nmrf == TRUE))
effect_sizes_nmrf$grouping <- "Molecular Phenotype"
effect_sizes_nmrf$grouping_nmrf <- "No MRF Pregnancies"

effect_sizes_combined <- bind_rows(effect_sizes_all, effect_sizes_nmrf, effect_sizes_mrf)
effect_sizes_combined$grouping_nmrf <- factor(effect_sizes_combined$grouping_nmrf, levels = c("All Pregnancies", "MRF Pregnancies", "No MRF Pregnancies"))
effect_sizes_combined$hdp_vanity_group <- factor(effect_sizes_combined$hdp_vanity_group,
  levels = c(
    "Control",
    "GHTN",
    "Postpartum PE",
    "Term PE,  Not Severe, Term Diagnosis",
    "Term PE, Not Severe, Preterm Diagnosis",
    "Term PE, Severe, Term Diagnosis",
    "Term PE, Severe, Preterm Diagnosis",
    "Preterm PE, Not Severe",
    "Preterm PE, Severe"
  )
)


# coarse groups
effect_sizes_coarse_all <- compute_hdp_effects_coarse(df)
effect_sizes_coarse_all$grouping <- "Molecular Phenotype"
effect_sizes_coarse_all$grouping_nmrf <- "All Pregnancies"

effect_sizes_coarse_mrf <- compute_hdp_effects_coarse(df %>% filter(is_mrf == TRUE))
effect_sizes_coarse_mrf$grouping <- "Maternal risk factor (RF)"
effect_sizes_coarse_mrf$grouping_nmrf <- "MRF Pregnancies"

effect_sizes_coarse_nmrf <- compute_hdp_effects_coarse(df %>% filter(is_nmrf == TRUE))
effect_sizes_coarse_nmrf$grouping <- "Maternal risk factor (RF)"
effect_sizes_coarse_nmrf$grouping_nmrf <- "No MRF Pregnancies"

effect_sizes_coarse_combined <- bind_rows(effect_sizes_coarse_all, effect_sizes_coarse_nmrf, effect_sizes_coarse_mrf)
effect_sizes_coarse_combined$grouping_nmrf <- factor(effect_sizes_coarse_combined$grouping_nmrf, levels = c("All Pregnancies", "MRF Pregnancies", "No MRF Pregnancies"))
effect_sizes_coarse_combined$hdp_severity <- factor(effect_sizes_coarse_combined$hdp_severity,
  levels = c("Non HDP", "Immune Driven", "Placenta Driven")
)
effect_sizes_coarse_combined$grouping <- factor(effect_sizes_coarse_combined$grouping,
                                                    levels = c("Molecular Phenotype", "Maternal risk factor (RF)")
)

# PE mean
reference_group <- df[df$hdp_severity == "control", ]$PAPPA2
effect_size_pe <- df %>%
  select(is_pe, hdp_vanity_group, PAPPA2) %>%
  group_by(is_pe) %>%
  summarise(
    effect = effectsize::cohens_d(PAPPA2, y = reference_group, adjust = FALSE),
    n = n()
  )


# Color map --------------------------------------------------------------------
top_effect <- 1.75
bottom_effect <- 0
raster_height <- 0.95
diverging_heat_colormap <- colorspace::scale_fill_continuous_divergingx(
  palette = "RdBu",
  mid = color_mean,
  rev = TRUE,
  limits = c(bottom_effect, top_effect + 0.15),
  h3 = 255,
  c1 = 10,
  c3 = 255,
  l1 = 10,
  l3 = 0,
  p1 = 0.8,
  p2 = 1.02,
  p3 = 0.5,
  p4 = 0.5
)

# 2a. Heatmap ------------------------------------------------------------------
color_mean <- effect_size_pe[effect_size_pe$is_pe == TRUE, ]$effect$Cohens_d
heatmap <- ggplot(
  effect_sizes_all,
  aes(x = grouping, y = hdp_vanity_group, fill = effect)
) +
  geom_tile(aes(width = 0.5, height = raster_height)) +
  scale_x_discrete(position = "top", expand = c(0, 0)) +
  theme_minimal() +
  diverging_heat_colormap +
  theme(
    legend.position = "none",
    panel.grid = element_blank(),
    axis.title = element_blank()
  ) +
  coord_cartesian(xlim = c(1, 1.05))

# 2b. Float Text NMRF ----------------------------------------------------------
combined_nmrf <- ggplot(effect_sizes_coarse_combined[effect_sizes_coarse_combined$hdp_severity != 'Non HDP', ], 
                        aes(x = grouping, y = effect, fill = effect)) +
  geom_segment(aes(x = as.numeric(grouping), xend = 2.4, group = grouping), 
               color = "gray20", 
               linetype = "dashed",
               alpha = 0.3) +
  ggtext::geom_richtext(aes(label = hdp_severity),
                        color = "white",
                        label.padding = unit(c(0.2,0.8,0.2,0.8), "lines"),
                        label.margin = unit(c(0, 0, 0, 0), "lines"),
                        hjust = 0.5, vjust = 0.5,
                        size = 3,
                        fontface = "bold") + 
  coord_cartesian(ylim = c(0, top_effect)) +
  theme_minimal() +
  scale_x_discrete(
    position = "top",
    labels = function(x) str_wrap(x, width = 10)
  ) +
  scale_y_continuous(breaks = seq(bottom_effect, top_effect, 0.25), minor_breaks = NULL) +
  diverging_heat_colormap +
  theme(
    panel.grid.major.x = element_blank(),
    axis.title.y = element_blank(),
    axis.text.y = element_blank(),
    axis.title.x = element_blank()
  ) +
  theme(legend.position = "none")

# 2b. color bar ----------------------------------------------------------------
color_top <- top_effect * 200
colorbar_grid <- data.frame(id = rep(1, color_top), effect = (1:color_top) / 200)
color_bar <- ggplot(colorbar_grid) +
  geom_tile(aes(x = 0.2, y = effect, fill = effect), width = 0.2) +
  scale_y_continuous(
    position = "right",
    breaks = seq(bottom_effect, top_effect, 0.25),
    minor_breaks = NULL
  ) +
  scale_x_continuous(limits = c(0, 0.4), breaks = 1) +
  coord_cartesian(ylim = c(0, top_effect)) +
  theme_minimal() +
  diverging_heat_colormap +
  theme(legend.position = "none", axis.title = element_blank())

# save figure ------------------------------------------------------------------
# Single heatmap
combined_fig <- heatmap + plot_spacer() + combined_nmrf + color_bar +
  plot_layout(widths = c(3, 6, 25, 2))
ggsave("fig2abc_combined.pdf", combined_fig, width = 8, height = 5.5)
ggsave("fig2abc_combined.svg", combined_fig, width = 9.5, height = 5.5)

library(tidyverse)
library(patchwork)

df <- arrow::read_feather("sample_data.feather")
df <- df[df$is_nmrf == TRUE, ]

# List of columns for which to calculate Cohen's d
columns <- c("PAPPA2", "CD163", "VSIG4", "ADAM12", "XAGE2", "KISS1", "PAPPA", "PGF")

# Function to calculate Cohen's d for a given column and pair of groups
cohens_d_groups_granular <- function(df, column, group1, group2) {
  effectsize::cohens_d(
    df[df$hdp_vanity_group == group1, column][[column]],
    df[df$hdp_vanity_group == group2, column][[column]]
  )
}
granular_groups <- c(
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

# Get group counts for table
group_counts <- df %>%
  select(hdp_vanity_group) %>%
  group_by(hdp_vanity_group) %>%
  summarise(n = n()) %>%
  arrange(desc(hdp_vanity_group))
group_counts$hdp_vanity_group <- factor(group_counts$hdp_vanity_group, levels = granular_groups)
group_counts %>% arrange(desc(hdp_vanity_group))

# Loop over groups and genes
results <- data.frame()
for (group in granular_groups) {
  group_ds <- sapply(columns, function(column) {
    cohens_d_groups_granular(df, column, group, "Control")[["Cohens_d"]]
  })
  sig <- sapply(columns, function(column) {
    sign(cohens_d_groups_granular(df, column, group, "Control")[["CI_low"]]) == sign(cohens_d_groups_granular(df, column, group, "Control")[["CI_high"]])
  })
  group_ds <- data.frame(effect = group_ds, sig = sig, gene = columns)
  group_ds$grouping <- group
  results <- bind_rows(results, group_ds)
}

# PE mean
reference_group <- df[df$hdp_vanity_group == "Control", ]$PAPPA2
effect_size_pe <- df %>%
  select(is_pe, hdp_vanity_group, PAPPA2) %>%
  group_by(is_pe) %>%
  summarise(
    effect = effectsize::cohens_d(PAPPA2, y = reference_group, adjust = FALSE),
    n = n()
  )

# Heatmap - granular
raster_height <- 0.95
results$grouping <- factor(results$grouping,
  levels = granular_groups
)
results$gene <- factor(results$gene,
  levels = columns
)
coarse_heatmap <- ggplot(
  results[results$gene %in% c("PAPPA2", "CD163", "VSIG4", "ADAM12", "XAGE2", "KISS1"), ],
  aes(x = gene, y = grouping, fill = effect)
) +
  geom_tile(aes(width = raster_height, height = raster_height)) +
  scale_x_discrete(position = "top", expand = c(0, 0)) +
  theme_minimal() +
  colorspace::scale_fill_continuous_divergingx(
    palette = "Spectral",
    mid = 0,
    rev = TRUE,
    limits = c(-0.5, 1.75),
    name = "Cohen's *d*",
    p1 = 0.6,
    p2 = 0.8,
    p4 = 0.6
  ) +
  # geom_tile(data = results[results$sig == FALSE, ], # to hide non-significant
  #  aes(width = raster_height, height = raster_height), fill = "white") +
  geom_vline(xintercept = 6.5) +
  theme(
    panel.grid = element_blank(),
    axis.title = element_blank(),
    legend.key.height = unit(2.5, "lines"),
    legend.key.width = unit(0.5, "lines"),
    legend.title = ggtext::element_markdown()
  )
ggsave("fig4c_heatmap_granular.pdf", coarse_heatmap, width = 7, height = 5)
ggsave("fig4c_heatmap_granular.svg", coarse_heatmap, width = 7, height = 5)

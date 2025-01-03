library(tidyverse)

# NMRF AMA LR+ values (from CV data results notebook)
classifier_LRpos <- 2.417301921285916
uspstf_LRpos <- 1.5086685292204065

classifier_LRpos_dga_lte35 <- 3.1968860424028276
uspstf_LRpos_dga_lte35 <- 1.6537128712871285

classifier_LRpos_AMA <- 3.4937611408199643
uspstf_LRpos_AMA <- 1.1684053651266764


# Plot results
barplot <- ggplot(
  df_lrplus,
  aes(
    x = subgroup,
    fill = subgroup,
    y = positive_likelihood_ratio,
    group = pred_source
  )
) +
  geom_bar(aes(alpha=pred_source), 
           position="dodge", stat="identity", show.legend = FALSE, width = 0.6) +
  geom_text(aes(label = positive_likelihood_ratio_rounded),
            vjust = -0.7, show.legend = FALSE,
            size = 5,
            position = ggplot2::position_dodge(width=0.6)
  ) +
  scale_y_continuous(breaks = c(1, 1.5, 2, 2.5, 3, 3.5, 4.0)) +
  coord_cartesian(ylim = c(1, 3.8), xlim = c(0.5, 3.5), expand = FALSE) +
  theme_minimal() +
  scale_fill_manual(values = c(
    "All NMRF" = "#1f77b4",
    "NMRF, delivery <= 35 weeks" = "#ff7f0e",
    "NMRF, AMA" = "#2ca02c"
  )) +
  scale_alpha_manual(values = c(0.4, 0.8)) +
  ylab("Positive Likelihood Ratio") +
  theme(
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    axis.title.x = element_blank()
  )
ggsave("fig5b_lrpos.pdf", barplot, width = 6, height = 4)
ggsave("fig5b_lrpos.svg", barplot, width = 6, height = 4)

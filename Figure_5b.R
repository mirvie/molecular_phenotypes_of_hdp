library(tidyverse)

# NMRF AMA LR+ values (from CV data results notebook)
classifier_LRpos <- 3.49376
uspstf_LRpos <- 1.16841

# Plot results
df_lrplus <- data.frame(
  positive_likelihood_ratio = c(classifier_LRpos, uspstf_LRpos),
  pred_source = c("Classifier", "USPSTF")
)
df_lrplus$positive_likelihood_ratio_rounded <- round(df_lrplus$positive_likelihood_ratio, 2)
barplot <- ggplot(
  df_lrplus,
  aes(
    x = pred_source,
    y = positive_likelihood_ratio,
    fill = pred_source
  )
) +
  geom_col(alpha = 0.9, show.legend = FALSE, width = 0.6) +
  geom_text(aes(label = positive_likelihood_ratio_rounded),
    vjust = -0.7, show.legend = FALSE,
    size = 5
  ) +
  scale_fill_manual(values = c(
    "Classifier" = "#ff7f0e",
    "USPSTF" = "#929292"
  )) +
  scale_y_continuous(breaks = c(1, 1.5, 2, 2.5, 3, 3.5)) +
  coord_cartesian(ylim = c(1, 3.8), xlim = c(0.5, 2.5), expand = FALSE) +
  theme_minimal() +
  ylab("Positive Likelihood Ratio") +
  theme(
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    axis.title.x = element_blank()
  )
ggsave("fig5b_lrpos.pdf", barplot, width = 6, height = 4)
ggsave("fig5b_lrpos.svg", barplot, width = 6, height = 4)

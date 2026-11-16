# ------------------------------
# MULTIVARIATE SUR ON IMPUTED DATASET (missForest + horseshoe prior)
# ------------------------------

library(readr)
library(dplyr)
library(missForest)
library(R2jags)
library(coda)
library(stringr)
library(ggplot2)
library(tidyr)
library(corrplot)
colnames(df_imputed)


# Load and prepare raw data
raw_df <- read_csv("Maire_Plant.csv") %>%
  select(-ID, -SITECODE) %>%  
  select(SLA, Aarea, Narea, Parea, Gs, PPTmean:Pavail) %>%  
  mutate(across(c(SLA, Aarea, Narea, Parea, Gs), log))

traits <- c("SLA", "Aarea", "Narea", "Parea", "Gs")
predictors <- setdiff(names(raw_df), traits)

# Impute missing values
set.seed(123)
imputed <- missForest(as.data.frame(raw_df))
df_imputed <- imputed$ximp

# b Save the missForest object
save(imputed, file = "missForest_Imputation_Output.Rdata")
cat("b Saved: missForest_Imputation_Output.Rdata\n")
load("missForest_Imputation_Output.Rdata")
# Save summary of imputed data to PDF
pdf("Imputed_Data_Summary.pdf", width = 8, height = 10)
plot.new()
text(0, 1, "Summary of Imputed Data", cex = 1.5, pos = 4)
summary_text <- capture.output(summary(df_imputed))
text_lines <- strwrap(summary_text, width = 110)
for (i in seq_along(text_lines)) {
  text(0, 1 - i * 0.05, text_lines[i], pos = 4, cex = 0.8)
}
dev.off()
cat("b Saved summary to Imputed_Data_Summary.pdf\n")


# Prepare data for SUR model
X <- scale(df_imputed[, predictors])
Y <- as.matrix(df_imputed[, traits])
n <- nrow(Y)
p <- ncol(X)
k <- ncol(Y)
R <- diag(k)

jags_data <- list(Y = Y, X = X, n = n, p = p, k = k, R = R, df = k + 1)

# JAGS model string
model_string_sur <- "
model {
  for (i in 1:n) {
    Y[i, 1:k] ~ dmnorm(mu[i,], Omega)
    for (j in 1:k) {
      mu[i,j] <- alpha[j] + inprod(X[i,], beta[j,])
    }
  }
  for (j in 1:k) {
    alpha[j] ~ dnorm(0, 0.01)
    for (m in 1:p) {
      beta[j,m] ~ dnorm(0, prec[j,m])
      prec[j,m] <- 1 / (lambda2[j,m] * tau_global2 * sigma2[j])
      lambda2[j,m] ~ dgamma(0.5, 1 / nu[j,m])
      nu[j,m] ~ dgamma(0.5, 1)
    }
    tau[j] ~ dgamma(2, 1)
    sigma[j] <- 1 / sqrt(tau[j])
    sigma2[j] <- sigma[j]^2
  }
  tau_global2 ~ dgamma(0.5, 1 / xi)
  xi ~ dgamma(0.5, 1)
  Omega ~ dwish(R, df)
  Sigma <- inverse(Omega)
}
"

# Run the JAGS model
model_SUR_imputed <- jags(
  data = jags_data,
  parameters.to.save = c("alpha", "beta", "Sigma", "mu"),
  model.file = textConnection(model_string_sur),
  n.chains = 1,
  
  n.iter = 2000,
  n.burnin = 500,
  n.thin = 5
)

# Save the fitted model
save(model_SUR_imputed, file = "model_SUR_missForest_CLEANED.Rdata")
cat("b Model saved to model_SUR_missForest_CLEANED.Rdata\n")


# MULTIVARIATE SUR ON COMPLETE CASES (Horseshoe Prior)

library(readr)
library(dplyr)
library(R2jags)
library(coda)
library(stringr)


# Load and prepare data
file_path <- "Maire_Plant.csv"
df <- read_csv(file_path) %>%
  select(SLA, Aarea, Narea, Parea, Gs, PPTmean:Pavail) %>%
  mutate(across(c(SLA, Aarea, Narea, Parea, Gs), log))

traits <- c("SLA", "Aarea", "Narea", "Parea", "Gs")
predictor_vars <- setdiff(names(df), traits)

# Subset complete cases
complete_df <- df[complete.cases(df[, traits]), ]

# Prepare data for JAGS
X <- scale(complete_df[, predictor_vars])
Y <- as.matrix(complete_df[, traits])
n <- nrow(Y)
p <- ncol(X)
k <- ncol(Y)
R <- diag(k)

jags_data <- list(Y = Y, X = X, n = n, p = p, k = k, R = R, df = k + 1)

# JAGS model string
model_string_sur <- "
model {
  for (i in 1:n) {
    Y[i, 1:k] ~ dmnorm(mu[i,], Omega)
    for (j in 1:k) {
      mu[i,j] <- alpha[j] + inprod(X[i,], beta[j,])
    }
  }
  for (j in 1:k) {
    alpha[j] ~ dnorm(0, 0.01)
    for (m in 1:p) {
      beta[j,m] ~ dnorm(0, prec[j,m])
      prec[j,m] <- 1 / (lambda2[j,m] * tau_global2 * sigma2[j])
      lambda2[j,m] ~ dgamma(0.5, 1 / nu[j,m])
      nu[j,m] ~ dgamma(0.5, 1)
    }
    tau[j] ~ dgamma(2, 1)
    sigma[j] <- 1 / sqrt(tau[j])
    sigma2[j] <- sigma[j]^2
  }
  tau_global2 ~ dgamma(0.5, 1 / xi)
  xi ~ dgamma(0.5, 1)
  Omega ~ dwish(R, df)
  Sigma <- inverse(Omega)
}
"

# Run JAGS
model_SUR_complete <- jags(
  data = jags_data,
  parameters.to.save = c("alpha", "beta", "Sigma"),
  model.file = textConnection(model_string_sur),
  n.chains = 1,
  n.iter = 5000,
  n.burnin = 1000,
  n.thin = 5
)

# Save model
save(model_SUR_complete, file = "SUR_Multi_Complete_Cases_Model.Rdata")
cat("b Model saved to SUR_Multi_Complete_Cases_Model.Rdata\n")



# UNIVARIATE REGRESSIONS PER TRAIT (Horseshoe Prior)

library(readr)
library(dplyr)
library(R2jags)
library(coda)
library(stringr)
library(ggplot2)library(mice)
traits <- c("SLA", "Aarea", "Narea", "Parea", "Gs")
md.pattern(df[, traits])

# -------------------------------
# Load and Prepare Data
# -------------------------------
file_path <- "Maire_Plant.csv"
df <- read_csv(file_path) %>%
  select(SLA, Aarea, Narea, Parea, Gs, PPTmean:Pavail) %>%
  mutate(across(c(SLA, Aarea, Narea, Parea, Gs), log))

traits <- c("SLA", "Aarea", "Narea", "Parea", "Gs")
predictor_vars <- setdiff(names(df), traits)

# -------------------------------
# JAGS Model Function
# -------------------------------
run_jags_model_for_trait <- function(df_trait, trait_name) {
  X <- scale(df_trait[, predictor_vars])
  Y <- df_trait[[trait_name]]
  jags_data <- list(X = X, Y = Y, n = nrow(X), p = ncol(X))
  
  model_string <- "
  model {
    for (i in 1:n) {
      Y[i] ~ dnorm(mu[i], tau)
      mu[i] <- alpha + inprod(X[i,], beta[])
    }
    alpha ~ dnorm(0, 0.01)
    for (j in 1:p) {
      beta[j] ~ dnorm(0, prec[j])
      prec[j] <- 1 / (lambda2[j] * tau_global2 * sigma2)
      lambda2[j] ~ dgamma(0.5, 1 / nu[j])
      nu[j] ~ dgamma(0.5, 1)
    }
    tau ~ dgamma(2, 1)
    sigma <- 1 / sqrt(tau)
    sigma2 <- sigma^2
    tau_global2 ~ dgamma(0.5, 1 / xi)
    xi ~ dgamma(0.5, 1)
  }
  "
  
  model_result <- jags(
    data = jags_data,
    parameters.to.save = c("alpha", "beta"),
    model.file = textConnection(model_string),
    n.chains = 1,
    n.iter = 5000,
    n.burnin = 1000,
    n.thin = 5
  )
  
  model_result$X_names <- colnames(X)
  return(model_result)
}

# -------------------------------
# Fit Univariate Models
# -------------------------------
model_uni_results <- list(
  SLA = run_jags_model_for_trait(df[!is.na(df$SLA), ], "SLA"),
  Aarea = run_jags_model_for_trait(df[!is.na(df$Aarea), ], "Aarea"),
  Narea = run_jags_model_for_trait(df[!is.na(df$Narea), ], "Narea"),
  Parea = run_jags_model_for_trait(df[!is.na(df$Parea), ], "Parea"),
  Gs = run_jags_model_for_trait(df[!is.na(df$Gs), ], "Gs")
)

# -------------------------------
# Credible Interval Plots: Combined (5-Panel without Intercept)
# -------------------------------
pdf("Univariate_CompleteCases_CredibleIntervals_Combined.pdf", width = 14, height = 7)
par(mfrow = c(1, 5), mar = c(4, 1, 3, 1), oma = c(0, 6, 0, 0))

for (trait in names(model_uni_results)) {
  model <- model_uni_results[[trait]]
  samples_matrix <- model$BUGSoutput$sims.matrix
  beta_cols <- grep("^beta\\[", colnames(samples_matrix), value = TRUE)
  
  beta_summary <- data.frame(
    Predictor = model$X_names,
    Mean = apply(samples_matrix[, beta_cols], 2, mean),
    Lower = apply(samples_matrix[, beta_cols], 2, function(x) quantile(x, 0.025)),
    Upper = apply(samples_matrix[, beta_cols], 2, function(x) quantile(x, 0.975))
  )
  beta_summary$Significant <- !(beta_summary$Lower < 0 & beta_summary$Upper > 0)
  n <- nrow(beta_summary)
  
  plot(
    beta_summary$Mean, 1:n,
    xlim = range(c(beta_summary$Lower, beta_summary$Upper)),
    ylim = c(1, n),
    yaxt = "n",
    pch = ifelse(beta_summary$Significant, 19, 1),
    xlab = "Coefficient",
    ylab = "",
    main = trait
  )
  segments(beta_summary$Lower, 1:n, beta_summary$Upper, 1:n)
  abline(v = 0, col = "red", lty = 2)
  if (trait == names(model_uni_results)[1]) {
    axis(2, at = 1:n, labels = beta_summary$Predictor, las = 1, cex.axis = 0.7)
  }
}
dev.off()
cat("b Saved: Univariate_CompleteCases_CredibleIntervals_Combined.pdf\n")

# -------------------------------
# Barplot: Number of Significant Predictors per Trait
# -------------------------------
sig_summary_list <- lapply(names(model_uni_results), function(trait) {
  sims <- model_uni_results[[trait]]$BUGSoutput$sims.matrix
  beta_samples <- sims[, grep("^beta\\[", colnames(sims), value = TRUE)]
  lower <- apply(beta_samples, 2, function(x) quantile(x, 0.025))
  upper <- apply(beta_samples, 2, function(x) quantile(x, 0.975))
  significant <- !(lower < 0 & upper > 0)
  data.frame(Trait = trait, Significant = sum(significant))
})

sig_df_uni <- do.call(rbind, sig_summary_list)
sig_df_uni$Model <- "Univariate Complete Cases"

pdf("Significant_Predictors_Univariate.pdf", width = 8, height = 5)
ggplot(sig_df_uni, aes(x = Trait, y = Significant, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Number of Significant Predictors per Trait (Univariate)",
       y = "Count", x = "Trait") +
  theme_minimal() +
  theme(text = element_text(size = 12), legend.position = "none")
dev.off()
cat("b Saved: Significant_Predictors_Univariate.pdf\n")


# Save all univariate models to a single RData file
save(model_uni_results, file = "Univariate_Models_CompleteCases.Rdata")
cat("b Saved all univariate models to Univariate_Models_CompleteCases.Rdata\n")
save(model_uni_results, file = "Univariate_Models_CompleteCases.Rdata")



library(readr)
library(dplyr)
library(knitr)
library(kableExtra)

# Load raw data
file_path <- "Maire_Plant.csv"

raw_df <- read_csv(file_path) %>%
  select(SLA, Aarea, Narea, Parea, Gs, PPTmean:Pavail)

# Define traits
traits <- c("SLA", "Aarea", "Narea", "Parea", "Gs")

# Compute summary stats
summary_stats <- data.frame(
  Trait = traits,
  Mean = sapply(traits, function(t) mean(raw_df[[t]], na.rm = TRUE)),
  `Standard Deviation` = sapply(traits, function(t) sd(raw_df[[t]], na.rm = TRUE)),
  `Proportion Missing` = sapply(traits, function(t) mean(is.na(raw_df[[t]])))
)

# Print nicely formatted table
kbl(summary_stats, digits = 3,
    caption = "Summary statistics for response traits") %>%
  kable_styling(latex_options = c("striped", "hold_position", "scale_down"),
                full_width = FALSE, position = "center")

# Required library
library(mice)

# Load data
df <- read.csv("Maire_Plant.csv")
traits <- c("SLA", "Aarea", "Narea", "Parea", "Gs")

# Save the md.pattern plot
png("md_pattern_missingness.png", width = 800, height = 600)
md.pattern(df[, traits], rotate.names = TRUE)
dev.off()
# Load imputed dataset
load("missForest_Imputation_Output.Rdata")
df_imputed <- imputed$ximp

# 0. md.pattern-style Missingness Heatmap (Styled)
missing_data <- read.csv("Maire_Plant.csv")
traits <- c("SLA", "Aarea", "Narea", "Parea", "Gs")

png("Figures/Figure0_Missingness_Heatmap.png", width = 900, height = 700)
md.pattern(missing_data[, traits], rotate.names = TRUE)
dev.off()


library(ggplot2)
library(corrplot)
library(gridExtra)
library(vcd)
library(ggfortify)
library(patchwork)
library(dplyr)

# Load and subset the dataset
raw_df <- read.csv("Maire_Plant.csv") |>
  select(SLA, Aarea, Narea, Parea, Gs, PPTmean:Pavail) |>
  mutate(across(c(SLA, Aarea, Narea, Parea, Gs), log))

traits <- c("SLA", "Aarea", "Narea", "Parea", "Gs")
predictors <- setdiff(names(raw_df), traits)

# Create folders
dir.create("Figures", showWarnings = FALSE)
dir.create("Tables", showWarnings = FALSE)

# 1. Density Plots for Skewness
density_plots <- lapply(traits, function(trait) {
  ggplot(raw_df, aes_string(x = trait)) +
    geom_density(fill = "#0072B2", alpha = 0.7, na.rm = TRUE) +
    theme_minimal(base_size = 14) +
    labs(title = paste("Density of", trait), x = trait, y = "Density")
})
png("Figures/Skewness_Densities.png", 1400, 900, res = 150)
do.call(grid.arrange, c(density_plots, ncol = 3))
dev.off()

# 2. Trait Correlation Matrix
cor_matrix <- cor(raw_df[, traits], use = "pairwise.complete.obs")
png("Figures/Correlation_Matrix.png", 800, 800, res = 150)
corrplot(cor_matrix, method = "color", type = "upper",
         tl.col = "black", tl.cex = 1.2,
         addCoef.col = "black", number.cex = 1.0,
         title = "Trait Correlation Matrix", mar = c(0,0,2,0))
dev.off()

# 3. Missingness Boxplots
miss_df <- is.na(raw_df[, traits])
colnames(miss_df) <- paste0("miss_", traits)

boxplot_list <- list()
for (trait in traits) {
  for (miss_trait in traits) {
    if (trait != miss_trait) {
      boxplot_list[[paste(trait, miss_trait)]] <- ggplot(raw_df, aes(x = factor(is.na(raw_df[[miss_trait]])), y = raw_df[[trait]])) +
        geom_boxplot(na.rm = TRUE, fill = "#56B4E9") +
        labs(title = paste(trait, "by Missing", miss_trait), x = paste("Missing", miss_trait), y = trait) +
        theme_minimal(base_size = 13)
    }
  }
}
png("Figures/Boxplot_Trait_Missingness.png", 1400, 900, res = 150)
gridExtra::grid.arrange(
  boxplot_list[["SLA Parea"]],
  boxplot_list[["Narea Gs"]],
  boxplot_list[["Parea SLA"]],
  ncol = 2
)
dev.off()

# 4. Missingness Histograms
hist_list <- list()
for (trait in traits) {
  for (miss_trait in traits) {
    if (trait != miss_trait) {
      df_temp <- raw_df
      df_temp$missing_flag <- is.na(df_temp[[miss_trait]])
      hist_list[[paste(trait, miss_trait)]] <- ggplot(df_temp, aes(x = .data[[trait]], fill = missing_flag)) +
        geom_histogram(position = "identity", alpha = 0.6, bins = 30) +
        scale_fill_manual(values = c("TRUE" = "#D55E00", "FALSE" = "#009E73")) +
        theme_minimal(base_size = 13) +
        labs(title = paste(trait, "~ Missing", miss_trait), fill = "Missing?")
    }
  }
}
png("Figures/Histogram_Trait_Missingness.png", 1400, 900, res = 150)
gridExtra::grid.arrange(
  hist_list[["SLA Parea"]],
  hist_list[["Narea Gs"]],
  hist_list[["Parea SLA"]],
  ncol = 2
)
dev.off()

# 5. Mosaic Plot for Missingness Structure
missing_matrix <- as.data.frame(miss_df)
png("Figures/Mosaic_Missingness.png", 1000, 700, res = 150)
mosaic(~ miss_SLA + miss_Aarea + miss_Narea + miss_Parea + miss_Gs, data = missing_matrix,
       shade = TRUE, legend = TRUE,
       labeling_args = list(set_varnames = c()))
dev.off()

# 6. PCA Biplot
load("missForest_Imputation_Output.Rdata")
df_imputed <- imputed$ximp
pca <- prcomp(df_imputed[, predictors], scale. = TRUE)
png("Figures/PCA_biplot.png", 1200, 900, res = 150)
autoplot(pca, data = df_imputed, loadings = TRUE, loadings.label = TRUE) +
  ggtitle("PCA Biplot of Environmental Covariates") +
  theme_minimal(base_size = 14)
dev.off()
write.csv(pca$rotation[, 1:3], "Tables/PCA_Loadings.csv")


# Libraries
library(ggplot2)
library(ggrepel)
library(readr)
library(dplyr)

# Use imputed data you already created
# Ensure 'df_imputed' and 'predictors' exist from your workflow

# PCA on predictors (from imputed data)
pca <- prcomp(df_imputed[, predictors], scale. = TRUE)

# PCA scores and loadings
scores <- as.data.frame(pca$x)
loadings <- as.data.frame(pca$rotation)
loadings$Variable <- rownames(loadings)

# Variance explained
explained_var <- round(100 * summary(pca)$importance[2, 1:2], 2)

# PCA biplot (cleaned)
ggplot(scores, aes(x = PC1, y = PC2)) +
  geom_point(size = 1, alpha = 0.6) +
  geom_segment(data = loadings, aes(x = 0, y = 0, xend = PC1 * 5, yend = PC2 * 5),
               arrow = arrow(length = unit(0.2, "cm")), color = "red", alpha = 0.8) +
  geom_text_repel(data = loadings, aes(x = PC1 * 5, y = PC2 * 5, label = Variable),
                  color = "red", size = 4) +
  labs(title = "PCA Biplot of Environmental Covariates",
       x = paste0("PC1 (", explained_var[1], "%)"),
       y = paste0("PC2 (", explained_var[2], "%)")) +
  theme_minimal(base_size = 14)

# Save plot
ggsave("PCA_Biplot_Cleaned.rdata", width = 8, height = 6, dpi = 300)

# Save table of loadings
loadings_out <- loadings[, c("Variable", "PC1", "PC2", "PC3")]
write.csv(loadings_out, "PCA_Loadings_Table.csv", row.names = FALSE)
cat("✅ PCA biplot and loadings saved.\n")



save(missing_pattern, file = "missingness_pattern.rdata")
library(png)

# Define filenames and objects
plot_files <- c(
  "Figures/logo.png",
  "Figures/missingness_pattern.png",
  "generalised_pairs_plot.png",
  "Figures/EDA_Trait_Covariate_Scatterplots_4Traits.png",
  "PCA_Loadings_BarPlot.png",
  "PCA_Loadings_Heatmap.png",
  "univariate_models.png",
  "sur_complete_model.png",
  "sur_imputed_model.png"
)

# Loop through each and save as .rdata
for (file in plot_files) {
  img <- readPNG(file)
  
  # Create a valid R object name (optional: remove slashes and file extension)
  obj_name <- gsub("\\.|/|-", "_", tools::file_path_sans_ext(file))
  
  # Save the image as an .RData file
  save(img, file = paste0(obj_name, ".rdata"))
}
load("PCA_Loadings_Heatmap.rdata")
# Now `img` is available in your environment
grid::grid.raster(img)  # To display PNG image




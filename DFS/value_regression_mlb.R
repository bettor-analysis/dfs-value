library(readr)
library(dplyr)
library(ggplot2)
library(Metrics)
library(randomForest)
library(xgboost)
library(tibble)

# -----------------------------
# LOAD & CLEAN DATA
# -----------------------------
df <- read_csv("data/rw-mlb-player-pool.csv")
df_clean <- df %>% filter(FPTS > 0)

# -----------------------------
# CHECK CORRELATION BETWEEN FPTS AND SAL
# -----------------------------
correlation <- cor(df_clean$FPTS, df_clean$SAL, use = "complete.obs")

# -----------------------------
# LINEAR MODEL
# -----------------------------
model_linear <- lm(SAL ~ FPTS, data = df_clean)
df_clean <- df_clean %>%
  mutate(PredLinear = predict(model_linear, newdata = df_clean))

# -----------------------------
# LOG-LINEAR MODEL
# -----------------------------
model_log <- lm(log(SAL) ~ FPTS, data = df_clean)
df_clean <- df_clean %>%
  mutate(PredLog = exp(predict(model_log, newdata = df_clean)))

# -----------------------------
# POLYNOMIAL MODEL
# -----------------------------
model_poly <- lm(SAL ~ poly(FPTS, 2), data = df_clean)
df_clean <- df_clean %>%
  mutate(PredPoly = predict(model_poly, newdata = df_clean))

# -----------------------------
# RANDOM FOREST MODEL
# -----------------------------
model_rf <- randomForest(SAL ~ FPTS, data = df_clean, ntree = 500, importance = TRUE)
df_clean <- df_clean %>%
  mutate(PredRF = predict(model_rf, newdata = df_clean))

# -----------------------------
# XGBOOST MODEL
# -----------------------------
# Create model matrix (excluding intercept)
X <- as.matrix(df_clean[ , "FPTS"])  # use only FPTS
y <- df_clean$SAL

# Remove NAs just in case
valid_rows <- complete.cases(X, y)
X <- X[valid_rows, , drop = FALSE]
y <- y[valid_rows]

# Train model
model_xgb <- xgboost(data = X, label = y, nrounds = 100,
                     objective = "reg:squarederror", verbose = 0)

# Predict (for full df_clean — using full FPTS column)
df_clean$PredXGB <- predict(model_xgb, newdata = as.matrix(df_clean$FPTS))

# -----------------------------
# MODEL PERFORMANCE METRICS
# -----------------------------
mae <- function(actual, predicted) mean(abs(actual - predicted))
mape <- function(actual, predicted) mean(abs((actual - predicted) / actual)) * 100

# Linear
rmse_linear <- rmse(df_clean$SAL, df_clean$PredLinear)
mae_linear  <- mae(df_clean$SAL, df_clean$PredLinear)
mape_linear <- mape(df_clean$SAL, df_clean$PredLinear)
r2_linear   <- summary(model_linear)$adj.r.squared

# Log-Linear
rmse_log <- rmse(df_clean$SAL, df_clean$PredLog)
mae_log  <- mae(df_clean$SAL, df_clean$PredLog)
mape_log <- mape(df_clean$SAL, df_clean$PredLog)
r2_log   <- summary(model_log)$adj.r.squared

# Polynomial
rmse_poly <- rmse(df_clean$SAL, df_clean$PredPoly)
mae_poly  <- mae(df_clean$SAL, df_clean$PredPoly)
mape_poly <- mape(df_clean$SAL, df_clean$PredPoly)
r2_poly   <- summary(model_poly)$adj.r.squared

# Random Forest
rmse_rf <- rmse(df_clean$SAL, df_clean$PredRF)
mae_rf  <- mae(df_clean$SAL, df_clean$PredRF)
mape_rf <- mape(df_clean$SAL, df_clean$PredRF)
r2_rf   <- 1 - sum((df_clean$SAL - df_clean$PredRF)^2) / sum((df_clean$SAL - mean(df_clean$SAL))^2)

# XGBoost
rmse_xgb <- rmse(df_clean$SAL, df_clean$PredXGB)
mae_xgb  <- mae(df_clean$SAL, df_clean$PredXGB)
mape_xgb <- mape(df_clean$SAL, df_clean$PredXGB)
r2_xgb   <- 1 - sum((df_clean$SAL - df_clean$PredXGB)^2) / sum((df_clean$SAL - mean(df_clean$SAL))^2)

# -----------------------------
# SUMMARY TABLE
# -----------------------------
performance_summary <- tibble(
  Model = c("Linear", "Log-Linear", "Polynomial", "Random Forest", "XGBoost"),
  RMSE = c(rmse_linear, rmse_log, rmse_poly, rmse_rf, rmse_xgb),
  MAE  = c(mae_linear, mae_log, mae_poly, mae_rf, mae_xgb),
  MAPE = c(mape_linear, mape_log, mape_poly, mape_rf, mape_xgb),
  R2_Adj = c(r2_linear, r2_log, r2_poly, r2_rf, r2_xgb)
)

cat("\nModel Performance Summary:\n")
print(performance_summary)

# -----------------------------
# VISUALIZATION (Optional)
# -----------------------------
ggplot(df_clean, aes(x = FPTS, y = SAL)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "lm", se = FALSE, color = "blue") +
  labs(title = "Actual Salary vs Projected Points")

df_clean %>%
  mutate(resid_linear = model_linear$residuals) %>%
  ggplot(aes(x = FPTS, y = resid_linear)) +
  geom_point(alpha = 0.5) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  labs(title = "Residuals: Linear Model")

# -----------------------------
# VALUE GRADING (Choose Your Model Below)
# -----------------------------
df_clean <- df_clean %>%
  mutate(
    PredictedSalary = PredXGB,  # Swap for PredRF, PredXGB, etc.
    ValueDelta = PredictedSalary - SAL,
    Grade = case_when(
      ValueDelta >= 600 ~ "A | Great Value",
      ValueDelta >= 200 ~ "B | Good Value",
      ValueDelta > -200 ~ "C | Moderate Value",
      TRUE ~ "X | Fade"
    ),
    Grade = factor(Grade, levels = c("A | Great Value", "B | Good Value", "C | Moderate Value", "X | Fade"))
  ) %>%
  arrange(Grade, desc(ValueDelta))

# -----------------------------
# GRADE COUNTS
# -----------------------------
grade_counts <- df_clean %>%
  group_by(Grade) %>%
  summarise(Count = n(), .groups = 'drop')

print(grade_counts)

# -----------------------------
# FILTER FOR SPECIFIC PLAYER
# -----------------------------
player_name <- "Juan Soto"  # Change to desired player name

player_data <- df_clean %>%
  filter(PLAYER == player_name)



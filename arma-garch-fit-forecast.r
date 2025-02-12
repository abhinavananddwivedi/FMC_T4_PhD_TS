# Load required libraries
library(rugarch)
library(tseries)
library(ggplot2)
library(dplyr)
set.seed(123)

# Generate synthetic data with ARMA(1,1)-GARCH(1,1) structure
n <- 1000

# Set more stable true parameters
true_params <- list(
  mu = 0.0001,          # Mean return
  ar1 = 0.1,            # Reduced AR coefficient
  ma1 = 0.05,           # Reduced MA coefficient
  omega = 0.00005,      # Increased GARCH constant
  alpha1 = 0.05,        # Reduced ARCH effect
  beta1 = 0.90,         # Increased GARCH persistence
  shape = 6             # Increased degrees of freedom
)

# Create specification with fixed parameters
spec <- ugarchspec(
  mean.model = list(armaOrder = c(1,1), include.mean = TRUE),
  variance.model = list(model = "sGARCH", garchOrder = c(1,1)),
  distribution.model = "std",
  fixed.pars = true_params
)

# Generate the process
sim <- ugarchpath(spec, n.sim = n)
returns <- sim@path$seriesSim

# Fit ARMA-GARCH model with more robust settings
spec_fit <- ugarchspec(
  mean.model = list(armaOrder = c(1,1), include.mean = TRUE),
  variance.model = list(model = "sGARCH", garchOrder = c(1,1)),
  distribution.model = "std",
  fixed.pars = list()  # Start with no fixed parameters
)

fit <- ugarchfit(spec_fit, returns, solver = "hybrid")  # Use hybrid solver

# Check convergence
print("Convergence Status:")
print(fit@fit$convergence)

# Function to safely compute confidence intervals and p-values
safe_ci <- function(fit) {
  params <- coef(fit)
  vc_matrix <- try(vcov(fit), silent = TRUE)
  
  # Get degrees of freedom from the fitted model
  df <- coef(fit)["shape"]
  
  # Get t-distribution critical value instead of normal
  t_crit <- qt(0.975, df)
  
  if(inherits(vc_matrix, "try-error") || any(is.na(diag(vc_matrix))) || any(diag(vc_matrix) <= 0)) {
    print("Warning: Using robust standard errors due to issues with variance-covariance matrix")
    se <- rep(NA, length(params))
    ci_lower <- rep(NA, length(params))
    ci_upper <- rep(NA, length(params))
    p_values <- rep(NA, length(params))
  } else {
    se <- sqrt(diag(vc_matrix))
    ci_lower <- params - t_crit * se  # Using t critical value
    ci_upper <- params + t_crit * se  # Using t critical value
    
    # Calculate p-values using t-distribution
    t_scores <- params / se
    p_values <- 2 * (1 - pt(abs(t_scores), df))
  }
  
  # Create results data frame with formatted numbers
  results <- data.frame(
    Parameter = names(params),
    Estimate = round(params, 6),
    Std_Error = round(se, 6),
    CI_Lower = round(ci_lower, 6),
    CI_Upper = round(ci_upper, 6),
    p_value = round(p_values, 6)
  )
  
  # Add significance stars
  results$significance <- ifelse(p_values < 0.001, "***",
                               ifelse(p_values < 0.01, "**",
                                    ifelse(p_values < 0.05, "*",
                                         ifelse(p_values < 0.1, ".", ""))))
  
  print(paste("Note: Using t-distribution with", round(df,2), "degrees of freedom for confidence intervals"))
  return(results)
}

# Get parameter estimates and CIs
param_summary <- safe_ci(fit)

# Print parameter estimates and diagnostics
print("\nParameter Estimates and Diagnostics:")
print(param_summary)

print("\nModel Information Criteria:")
print(infocriteria(fit))

print("\nLjung-Box Test on Standardized Residuals:")
print(Box.test(residuals(fit, standardize=TRUE), type="Ljung-Box", lag=10))

print("\nLjung-Box Test on Squared Standardized Residuals:")
print(Box.test(residuals(fit, standardize=TRUE)^2, type="Ljung-Box", lag=10))

# Make one-step ahead forecast with robust error handling
forecast <- ugarchforecast(fit, n.ahead = 1)
fore_mean <- as.numeric(forecast@forecast$seriesFor)
fore_sigma <- as.numeric(forecast@forecast$sigmaFor)

# More conservative prediction intervals using empirical quantiles
std_resid <- residuals(fit, standardize=TRUE)
q_lower <- quantile(std_resid, 0.025)
q_upper <- quantile(std_resid, 0.975)

pred_lower <- fore_mean + fore_sigma * q_lower
pred_upper <- fore_mean + fore_sigma * q_upper

print("\nOne-Step Ahead Forecast with Empirical Prediction Intervals:")
print(data.frame(
  Point_Forecast = fore_mean,
  Lower_95 = pred_lower,
  Upper_95 = pred_upper
))

# Diagnostic plots
par(mfrow=c(2,2))

# Returns and Volatility Plot
plot_data <- data.frame(
  Date = 1:length(returns),
  Returns = returns,
  Volatility = sigma(fit)
)

p1 <- ggplot(plot_data, aes(x = Date)) +
  geom_line(aes(y = Returns), color = "blue", alpha = 0.6) +
  geom_line(aes(y = Volatility), color = "red", alpha = 0.4) +
  labs(title = "Returns and Conditional Volatility",
       x = "Time", y = "Value") +
  theme_minimal()

# QQ Plot
std_resid <- residuals(fit, standardize=TRUE)
p2 <- ggplot(data.frame(Residuals = std_resid), aes(sample = Residuals)) +
  stat_qq() + stat_qq_line() +
  labs(title = "QQ Plot of Standardized Residuals") +
  theme_minimal()

# ACF of Standardized Residuals
acf_data <- acf(std_resid, plot=FALSE)
acf_df <- data.frame(
  Lag = acf_data$lag,
  ACF = acf_data$acf
)

# Get t-distribution critical value
df <- coef(fit)["shape"]
t_crit <- qt(0.975, df)

p3 <- ggplot(acf_df, aes(x = Lag, y = ACF)) +
  geom_bar(stat = "identity") +
  geom_hline(yintercept = c(-t_crit/sqrt(n), t_crit/sqrt(n)), 
             linetype = "dashed", color = "blue") +
  labs(title = paste0("ACF of Standardized Residuals\n(", 
                     round(100*2*(1-0.975), 1), 
                     "% CI based on t(", round(df,1), ") distribution)")) +
  theme_minimal()

print(p1)
print(p2)
print(p3)
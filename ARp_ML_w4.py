import numpy as np
from scipy.optimize import minimize
from scipy.stats import t
from loglike_ARp_Norm_w4 import simulate_arp

# Define the log-likelihood function for AR(p) model
def logLikeARpNorm(x, y, p, const):
    penalizedLikelihood = -1e10  # Penalize invalid likelihoods

    # Extract theta and sigma_u from x
    theta = x[:const + p]  # Parameters: constant, trend (if any), and AR coefficients
    sigma_u = x[const + p]  # Standard deviation of error term

    if sigma_u <= 0:  # Ensure sigma_u is positive
        return penalizedLikelihood

    T = len(y)
    # Build lagged matrix for predictors with the correct ordering for coefficients
    Y = np.column_stack([y[p - i - 1:T - i - 1] for i in range(p)])

    # Add constant or trend if specified
    if const == 1:
        Y = np.column_stack((np.ones(T - p), Y))
    elif const == 2:
        trend = np.arange(1, T - p + 1).reshape(-1, 1)
        Y = np.column_stack((np.ones(T - p), trend, Y))

    # Trim initial observations for dimensional match
    y_trimmed = y[p:]

    # Calculate residuals
    uhat = y_trimmed - Y @ theta
    utu = uhat.T @ uhat  # Sum of squared residuals

    # Calculate the conditional log-likelihood
    log_likelihood = -0.5 * (T - p) * np.log(2 * np.pi) \
                     - 0.5 * (T - p) * np.log(sigma_u ** 2) \
                     - utu / (2 * sigma_u ** 2)

    # Penalize if log-likelihood is invalid
    if np.isnan(log_likelihood) or np.isinf(log_likelihood) or not np.isreal(log_likelihood):
        return penalizedLikelihood

    return log_likelihood

# Define the AR(p) MLE function
def ARpML(y, p, const, alpha=0.05):
    # Initialize parameter vector
    init_theta = np.zeros(const + p)  # Coefficients (AR and possibly constant/trend)
    init_sigma_u = np.std(y)  # Initial guess for sigma_u
    init_params = np.append(init_theta, init_sigma_u)

    # Define the negative log-likelihood as the objective
    def neg_log_likelihood(params):
        return -logLikeARpNorm(params, y, p, const)

    # Perform optimization using BFGS
    result = minimize(neg_log_likelihood, init_params, method='BFGS')

    # Check if optimization was successful
    if not result.success:
        print("Warning: Optimization did not converge.")
        
    # Extract MLE estimates
    mle_params = result.x
    theta_hat = mle_params[:-1]  # Estimated parameters
    sigma_u_hat = mle_params[-1]  # Estimated sigma_u

    # Calculate standard errors using the inverse Hessian
    hessian_inv = result.hess_inv  # Inverse of the Hessian (asymptotic covariance matrix)
    standard_errors = np.sqrt(np.diag(hessian_inv))

    # Separate standard errors for coefficients and sigma_u
    sd_theta_hat = standard_errors[:-1]
    sd_sigma_u_hat = standard_errors[-1]

    # Effective sample size
    T_eff = len(y) - p

    # Calculate t-statistics
    t_statistics = theta_hat / sd_theta_hat
    
    # Calculate p-values using the PDF (similar to MATLAB code)
    p_values = t.pdf(t_statistics, df=T_eff - p)  # Probability density at the observed t-statistics

    # Calculate confidence intervals for parameters
    critical_value = t.ppf(1 - alpha / 2, df=T_eff - p)
    conf_intervals = np.column_stack([
        theta_hat - critical_value * sd_theta_hat,
        theta_hat + critical_value * sd_theta_hat
    ])

    # Prepare output
    ML_results = {
        "T_eff": T_eff,
        "thetahat": theta_hat,
        "sigma_u_hat": sigma_u_hat,
        "sd_thetahat": sd_theta_hat,
        "sd_sigma_u_hat": sd_sigma_u_hat,
        "tstat": t_statistics,
        "pvalues": p_values,
        "theta_ci": conf_intervals,
        "log_likelihood": -result.fun,
        "success": result.success
    }

    return ML_results

# Example usage with simulated AR(p) data
n = 50000  # Number of data points
p = 4  # Number of lags
coefficients = [0.5, -0.3, 0.2, -0.1]  # AR coefficients
constant = 1.0  # Constant term
eps_std = 0.5  # Standard deviation of eps

# Simulate AR(p) data
y = simulate_arp(n, coefficients, constant=constant, eps_std=eps_std)

# Run the MLE function
ML_results = ARpML(y, p, const=1)

# Print the MLE results
print("Effective Sample Size (T_eff):", ML_results["T_eff"])
print("MLE Parameters (thetahat):", ML_results["thetahat"])
print("Estimated sigma_u (sigma_u_hat):", ML_results["sigma_u_hat"])
print("Standard Errors (sd_thetahat):", ML_results["sd_thetahat"])
print("Standard Error of sigma_u (sd_sigma_u_hat):", ML_results["sd_sigma_u_hat"])
print("t-Statistics (tstat):", ML_results["tstat"])
print("p-Values (pvalues):", np.array2string(ML_results["pvalues"], precision=6, floatmode='unique'))
print("Confidence Intervals (theta_ci):", ML_results["theta_ci"])
print("Log-Likelihood:", ML_results["log_likelihood"])
print("Optimization Success:", ML_results["success"])

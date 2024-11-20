import numpy as np
from scipy.optimize import minimize
from scipy.stats import t, laplace

# define log-likelihood function for AR(p) model with Laplace-distributed errors
def logLikeARpLaplace(x, y, p, const):
    penalizedLikelihood = -1e10  # penalize invalid likelihoods

    # extract AR parameters from x
    theta = x  # theta includes AR coefficients and possibly constant/trend

    T = len(y)
    # build lagged matrix for predictors
    Y = np.column_stack([y[i:T - p + i] for i in range(p)])  # lagged values

    # add constant or trend if specified
    if const == 1:
        Y = np.column_stack((np.ones(T - p), Y))
    elif const == 2:
        trend = np.arange(1, T - p + 1).reshape(-1, 1)
        Y = np.column_stack((np.ones(T - p), trend, Y))

    # trim initial observations to match dimensions
    y_trimmed = y[p:]

    # calculate residuals
    uhat = y_trimmed - Y @ theta  # ML residuals

    # calculate conditional log-likelihood for Laplace-distributed errors
    log_likelihood = -np.log(2) * (T - p) - np.sum(np.abs(uhat))

    # penalize if log-likelihood is invalid
    if np.isnan(log_likelihood) or np.isinf(log_likelihood) or not np.isreal(log_likelihood):
        return penalizedLikelihood

    return log_likelihood

# define AR(p) MLE function for Laplace noise
def ARpLaplaceMLE(y, p, const, alpha=0.05):
    # initialize parameter vector
    init_theta = np.zeros(const + p)  # coefficients (AR and possibly constant/trend)

    # define negative log-likelihood as objective
    def neg_log_likelihood(params):
        return -logLikeARpLaplace(params, y, p, const)

    # perform optimization using BFGS
    result = minimize(neg_log_likelihood, init_theta, method='BFGS')

    # check if optimization was successful
    if not result.success:
        print("Warning: Optimization did not converge.")
        
    # extract MLE estimates
    estimated_params = result.x
    theta_estimates = estimated_params  # estimated parameters

    # calculate standard errors using inverse Hessian
    hessian_inv = result.hess_inv  # inverse of Hessian (asymptotic covariance matrix)
    standard_errors = np.sqrt(np.diag(hessian_inv))

    # effective sample size
    T_eff = len(y) - p

    # calculate t-statistics and p-values for each parameter using PDF
    t_statistics = theta_estimates / standard_errors
    p_values = 2 * t.pdf(np.abs(t_statistics), df=T_eff - p)  # two-sided p-value using t-distribution PDF

    # calculate confidence intervals for parameters
    critical_value = t.ppf(1 - alpha / 2, df=T_eff - p)
    conf_intervals = np.column_stack([
        theta_estimates - critical_value * standard_errors,
        theta_estimates + critical_value * standard_errors
    ])

    # prepare output
    results = {
        "T_eff": T_eff,
        "theta_estimates": theta_estimates,
        "standard_errors": standard_errors,
        "t_statistics": t_statistics,
        "p_values": p_values,
        "confidence_intervals": conf_intervals,
        "log_likelihood": -result.fun,
        "optimization_success": result.success
    }

    return results

# function to simulate AR(p) data with Laplace noise
def simulate_arp_laplace(n, p, coefficients, constant, eps_scale):
    """
    Simulates an AR(p) process with a constant and Laplace-distributed errors.
    
    Parameters:
        n (int): Number of data points.
        p (int): Number of lags.
        coefficients (list): List of p coefficients for the AR(p) model.
        constant (float): Constant term in the AR(p) model.
        eps_scale (float): Scale parameter (b) for the Laplace noise (controls spread).
        
    Returns:
        np.ndarray: Simulated AR(p) time series.
    """
    y = np.zeros(n)
    eps = laplace.rvs(scale=eps_scale, size=n)  # Laplace-distributed noise
    
    # generate the AR(p) process
    for t in range(p, n):
        y[t] = constant + sum(coefficients[i] * y[t - i - 1] for i in range(p)) + eps[t]
        
    return y

# example usage with simulated AR(p) data
n = 50000  # number of data points
p = 4  # number of lags
coefficients = [0.5, -0.3, 0.2, -0.1]  # AR coefficients
constant = 1.0  # constant
eps_scale = np.sqrt(2) / 2  # scale parameter for Laplace noise (match variance of 2)

# simulate AR(p) data with Laplace noise
y = simulate_arp_laplace(n, p, coefficients, constant, eps_scale)

# run MLE function for Laplace noise
results = ARpLaplaceMLE(y, p, const=1)

# print MLE results
print("Effective Sample Size (T_eff):", results["T_eff"])
print("Estimated Parameters (theta_estimates):", results["theta_estimates"])
print("Standard Errors:", results["standard_errors"])
print("t-Statistics:", results["t_statistics"])
print("p-Values (p_values):", np.array2string(results["p_values"], precision=6, floatmode='unique'))
print("Confidence Intervals:", results["confidence_intervals"])
print("Log-Likelihood:", results["log_likelihood"])
print("Optimization Success:", results["optimization_success"])

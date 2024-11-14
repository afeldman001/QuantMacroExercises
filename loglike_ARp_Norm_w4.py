import numpy as np

def logLikeARpNorm(x, y, p, const):
    """
    Computes the conditional log-likelihood for a Gaussian AR(p) model.

    Parameters:
        x (np.ndarray): Parameter vector [c, d, theta_1, ..., theta_p, sigma_u].
        y (np.ndarray): Data vector of dimension T.
        p (int): Number of lags.
        const (int): 1 if constant, 2 if constant and linear trend in model.

    Returns:
        float: Log-likelihood value of the Gaussian AR(p) model.
    """
    penalizedLikelihood = -1e10  # Small value to penalize invalid likelihoods

    # Extract parameters from x
    theta = x[:const + p]  # [c, d, theta_1, ..., theta_p]
    sigma_u = x[const + p]  # Standard deviation of error term (sigma_u)

    # Ensure sigma_u is positive
    if sigma_u <= 0:
        return penalizedLikelihood

    # Sample size
    T = len(y)

    # Create matrix with lagged variables
    Y = np.column_stack([y[i:T - p + i] for i in range(p)])

    # Add constant or trend if specified
    if const == 1:
        Y = np.column_stack((np.ones(T - p), Y))  # Add constant
    elif const == 2:
        trend = np.arange(1, T - p + 1).reshape(-1, 1)
        Y = np.column_stack((np.ones(T - p), trend, Y))  # Add constant and trend

    # Remove initial observations to align dimensions
    y_trimmed = y[p:]

    # Calculate residuals
    uhat = y_trimmed - Y @ theta  # Residuals
    utu = uhat.T @ uhat  # Sum of squared residuals

    # Compute the conditional log-likelihood
    log_likelihood = -0.5 * (T - p) * np.log(2 * np.pi) \
                     - 0.5 * (T - p) * np.log(sigma_u ** 2) \
                     - utu / (2 * sigma_u ** 2)

    # Penalize if log-likelihood is invalid
    if np.isnan(log_likelihood) or np.isinf(log_likelihood) or not np.isreal(log_likelihood):
        return penalizedLikelihood

    return log_likelihood

def simulate_arp(n, coefficients, constant=0, eps_std=1.0):
    """
    Simulates an AR(p) process with optional constant.

    Parameters:
        n (int): Number of data points to simulate.
        coefficients (list or np.ndarray): List of AR coefficients [theta_1, ..., theta_p].
        constant (float): Constant term in AR(p) model.
        eps_std (float): Standard deviation of white noise.

    Returns:
        np.ndarray: Simulated AR(p) time series.
    """
    p = len(coefficients)
    y = np.zeros(n)
    eps = np.random.normal(0, eps_std, n)

    # Generate the AR(p) process
    for t in range(p, n):
        y[t] = constant + sum(coefficients[i] * y[t - i - 1] for i in range(p)) + eps[t]

    return y

# Example usage
if __name__ == "__main__":
    # Parameters for an AR(4) model with constant
    x = np.array([1.0, 0.5, -0.2, 0.1, 0.3, 1.0])  # Example values for [c, theta_1, ..., theta_4, sigma_u]
    p = 4  # Number of lags for AR(4) model
    const = 1  # Model with constant term

    # Simulate an AR(4) process
    coefficients = x[1:p + 1]  # Extract AR coefficients from x
    y = simulate_arp(100, coefficients, constant=x[0], eps_std=x[-1])

    # Calculate log-likelihood
    log_likelihood = logLikeARpNorm(x, y, p, const)
    print("Log-Likelihood:", log_likelihood)

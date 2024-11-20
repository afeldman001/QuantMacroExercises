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
    penalizedLikelihood = -1e10  # small value to penalize invalid likelihoods

    # extract parameters from x
    theta = x[:const + p]  # [c, d, theta_1, ..., theta_p]
    sigma_u = x[const + p]  # standard deviation of error term (sigma_u)

    # ensure sigma_u is positive
    if sigma_u <= 0:
        return penalizedLikelihood

    # sample size
    T = len(y)

    # create matrix with lagged variables
    Y = np.column_stack([y[i:T - p + i] for i in range(p)])

    # add constant or trend if specified
    if const == 1:
        Y = np.column_stack((np.ones(T - p), Y))  # constant
    elif const == 2:
        trend = np.arange(1, T - p + 1).reshape(-1, 1)
        Y = np.column_stack((np.ones(T - p), trend, Y))  # constant and trend

    # remove initial observations to align dimensions
    y_trimmed = y[p:]

    # calculate residuals
    uhat = y_trimmed - Y @ theta  # residuals
    utu = uhat.T @ uhat  # sum of squared residuals

    # compute conditional log-likelihood
    log_likelihood = -0.5 * (T - p) * np.log(2 * np.pi) \
                     - 0.5 * (T - p) * np.log(sigma_u ** 2) \
                     - utu / (2 * sigma_u ** 2)

    # penalize if log-likelihood is invalid
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

    # generate AR(p) process
    for t in range(p, n):
        y[t] = constant + sum(coefficients[i] * y[t - i - 1] for i in range(p)) + eps[t]

    return y

# example usage
if __name__ == "__main__":
    # parameters for AR(4) model with constant
    x = np.array([1.0, 0.5, -0.2, 0.1, 0.3, 1.0])  # Example values for [c, theta_1, ..., theta_4, sigma_u]
    p = 4  # lags AR(4)
    const = 1  # with constant term

    # simulate AR(4) process
    coefficients = x[1:p + 1]  # extract AR coefficients from x
    y = simulate_arp(100, coefficients, constant=x[0], eps_std=x[-1])

    # calculate log-likelihood
    log_likelihood = logLikeARpNorm(x, y, p, const)
    print("Log-Likelihood:", log_likelihood)

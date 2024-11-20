import numpy as np
from loglike_ARp_Norm_w4 import simulate_arp

def lagOrderSelectionARp(y, const, pmax, crit):
    """
    Determine optimal lag order for AR(p) model using AIC, SIC (BIC), or HQC.

    Parameters:
        y (np.ndarray): data vector of dimension T.
        const (int): 1 for constant; 2 for constant and linear trend in the model.
        pmax (int): max lag order to test.
        crit (str): criterion to use ('AIC', 'SIC', or 'HQC').

    Returns:
        int: recommended lag order according to the specified criterion.
        np.ndarray: sorted results of lag orders and their criterion values.
    """
    T = len(y)
    T_eff = T - pmax  # effective sample size after trimming

    # initialize storage for information criteria
    info_crit = np.full(pmax, np.nan)

    for p in range(1, pmax + 1):
        n = const + p  # number of freely estimated parameters

        # prepare lagged data matrix
        Y = y[p:]  # dependent variable
        X = np.column_stack([y[p - i - 1:T - i - 1] for i in range(p)])  #lagged predictors

        # add constant or trend if specified
        if const == 1:
            X = np.column_stack((np.ones(len(Y)), X))
        elif const == 2:
            trend = np.arange(1, len(Y) + 1).reshape(-1, 1)
            X = np.column_stack((np.ones(len(Y)), trend, X))

        # OLS estimation
        theta_hat = np.linalg.pinv(X.T @ X) @ (X.T @ Y)  # OLS/ML estimator
        residuals = Y - X @ theta_hat  # Residuals

        # variance of errors
        sigma2u_ml = residuals.T @ residuals / T_eff  # ML estimate of variance of errors
        # sigma2u_ols = residuals.T @ residuals / (T_eff - n)  # OLS estimate of variance of errors (commented out)

        # ML estimate of variance in criteria calculations
        sigma2u = sigma2u_ml  # toggle to sigma2u_ols if OLS estimate is desired

        # compute information criteria
        if crit == 'AIC':  # Akaike
            info_crit[p - 1] = np.log(sigma2u) + (2 / T_eff) * n
        elif crit == 'SIC':  # Schwartz (BIC)
            info_crit[p - 1] = np.log(sigma2u) + (np.log(T_eff) / T_eff) * n
        elif crit == 'HQC':  # Hannan–Quinn
            info_crit[p - 1] = np.log(sigma2u) + (2 * np.log(np.log(T_eff)) / T_eff) * n
        else:
            raise ValueError("Invalid criterion. Use 'AIC', 'SIC', or 'HQC'.")

    # find lag order that minimizes chosen criterion
    nlag = np.argmin(info_crit) + 1  # +1 because Python uses zero-based indexing

    # store and sort results
    results = np.column_stack((np.arange(1, pmax + 1), info_crit))
    results = results[results[:, 1].argsort()]  # Sort by criterion values

    return nlag, results

# example
if __name__ == "__main__":
    # simulate AR(p) data
    n = 200  # number of data points
    p_true = 4  # true number of lags
    coefficients = [0.5, -0.3, 0.2]  # true coefficients
    constant = 1.0  # constant
    eps_std = 0.03  # standard deviation of noise

    # generate AR(p) data
    y = simulate_arp(n, coefficients, constant=constant, eps_std=eps_std)

    # determine optimal lag order
    pmax = 10  # max lag order to test
    const = 1  # constant 
    crit = 'SIC'  # criterion to use ('AIC', 'SIC', or 'HQC')

    optimal_lag, results = lagOrderSelectionARp(y, const, pmax, crit)
    print(f"Optimal lag order according to {crit}: {optimal_lag}\n")
    print("Lag Order Selection Results (Sorted):")
    print(f"{'Lag':<5}{'Criterion Value':>20}")
    for lag, value in results:
        print(f"{int(lag):<5}{value:>20.6f}")

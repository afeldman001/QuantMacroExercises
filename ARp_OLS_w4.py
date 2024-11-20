import numpy as np
import pandas as pd
from scipy.stats import t

# load data from local CSV file into a pandas DataFrame (replace with different data if desired)
data_df = pd.read_csv('/Users/aaronfeldman/Desktop/Quant_macro/Quantitative-Macroeconomics/data/AR4.csv')

# convert column 'AR4;' to numeric values, handling non-numeric entries (adjust as needed for different data file)
y = pd.to_numeric(data_df['AR4;'].str.replace(';', ''), errors='coerce').dropna().values

def ARpOLS(y, p, const, alpha=0.05):
    # prepare lagged matrix for predictors in order
    n = len(y)
    X = np.column_stack([y[p - i - 1:n - i - 1] for i in range(p)]).astype(float)  # Lagged predictor matrix
    
    # define dependent variable `Y`, dropping first `p` values
    Y = y[p:]

    # ensure `Y` and `X` are consistently trimmed
    min_length = min(len(Y), len(X))
    Y = Y[:min_length]
    X = X[:min_length]

    # add constant or trend if specified
    if const == 1:  # constant only
        X = np.column_stack((np.ones(len(Y)), X))
    elif const == 2:  # constant and trend
        trend = np.arange(1, len(Y) + 1).reshape(-1, 1)
        X = np.column_stack((np.ones(len(Y)), trend, X))

    # check for constant columns (zero variance) in X and print variances (debugging)
    variances = np.var(X, axis=0)
    print("Column variances:", variances)
    if any(variances == 0):
        print("Warning: X contains constant columns, which may cause singular matrix issues.")

    # calculate OLS coefficients with regularization for matrix inversion
    ridge_term = 1e-8  # small regularization term
    XtX_inv = np.linalg.inv(X.T @ X + ridge_term * np.eye(X.shape[1]))  # (X'X + ridge*I)^-1
    theta_hat = XtX_inv @ (X.T @ Y)  # theta_hat = (X'X)^-1 X'Y

    # calculate residuals and standard deviation of errors
    residuals = Y - X @ theta_hat
    T_eff = len(Y)  # Effective sample size
    sig_u_hat = np.sqrt((residuals.T @ residuals) / (T_eff - X.shape[1]))  # residual std deviation

    # standard errors of theta_hat
    sd_theta_hat = np.sqrt(np.diag(sig_u_hat**2 * XtX_inv))  # standard errors of coefficients

    # calculate t-statistics and p-values
    t_statistics = theta_hat / sd_theta_hat
    p_values = 2 * (1 - t.cdf(np.abs(t_statistics), df=T_eff - X.shape[1]))

    # confidence intervals for theta_hat
    critical_t = t.ppf(1 - alpha / 2, df=T_eff - X.shape[1])
    theta_ci = np.column_stack((theta_hat - critical_t * sd_theta_hat, 
                                theta_hat + critical_t * sd_theta_hat))

    # structure output
    OLS = {
        "T_eff": T_eff,
        "thetahat": theta_hat,
        "sd_thetahat": sd_theta_hat,
        "tstat": t_statistics,
        "pvalues": p_values,
        "siguhat": sig_u_hat,
        "theta_ci": theta_ci,
        "resid": residuals
    }

    return OLS


# only execute this part if script is run directly
if __name__ == "__main__":
   
    # set parameters for the ARpOLS function
    p = 4  # lags
    const = 1  # 1 for constant, 2 for constant + trend, or 0 for no constant
    alpha = 0.05  # significance level

    # call ARpOLS function
    results = ARpOLS(y, p, const, alpha)

    # print output
    print("Effective Sample Size (T_eff):", results["T_eff"])
    print("OLS Estimates (thetahat):", results["thetahat"])
    print("Standard Errors (sd_thetahat):", results["sd_thetahat"])
    print("t-Statistics (tstat):", results["tstat"])
    print("p-Values (pvalues):", results["pvalues"])
    print("Residual Std Deviation (siguhat):", results["siguhat"])
    print("Confidence Intervals (theta_ci):", results["theta_ci"])
    print("Residuals (resid):", results["resid"])

import numpy as np
import pandas as pd
from scipy.stats import chi2
from loglike_ARp_Norm_w4 import simulate_arp
from ARp_OLS_w4 import ARpOLS
from LagOrderSelection_ARp_w5 import lagOrderSelectionARp

def portmanteau_test(infl, pmax=12, const=0, alpha=0.05):
    """
    Perform portmanteau test for residual autocorrelation on an inflation series.

    Parameters:
        infl (np.ndarray): Inflation series.
        pmax (int): Maximum lag order for lag order selection.
        const (int): Include constant (1) or constant and linear trend (2).
        alpha (float): Significance level.

    Returns:
        None
    """
    # determine optimal lag order using AIC
    phat, _ = lagOrderSelectionARp(infl, const, pmax, crit='AIC')

    # estimate AR(phat) and AR(1) models
    ols_ar_phat = ARpOLS(infl, phat, const, alpha)
    ols_ar_1 = ARpOLS(infl, 1, const, alpha)

    # residuals
    u_p = ols_ar_phat["resid"]
    u_1 = ols_ar_1["resid"]
    T_p = len(u_p)
    T_1 = len(u_1)

    # maximum number of lags for test
    h = phat + 10

    # compute variances
    gam_p = (u_p @ u_p) / T_p
    gam_1 = (u_1 @ u_1) / T_1

    # compute autocorrelations
    rho_p = np.array([1 / ((T_p - j) * gam_p) * (u_p[j:] @ u_p[:-j]) for j in range(1, h + 1)])
    rho_1 = np.array([1 / ((T_1 - j) * gam_1) * (u_1[j:] @ u_1[:-j]) for j in range(1, h + 1)])

    # compute test statistics
    Q_p = T_p * np.sum(rho_p ** 2)
    Q_1 = T_1 * np.sum(rho_1 ** 2)

    # critical values and p-values from chi-squared distribution
    Qpcrit_phat = chi2.ppf(1 - alpha, h - phat)
    Qpcrit_1 = chi2.ppf(1 - alpha, h - 1)
    Qpval_phat = chi2.sf(Q_p, h - phat)
    Qpval_1 = chi2.sf(Q_1, h - 1)

    # crint results
    print("\nPORTMANTEAU TEST")
    print("H0: No remaining residual autocorrelation\n")
    print("Test Statistic > Critical Value")
    reject_phat = Q_p > Qpcrit_phat
    reject_1 = Q_1 > Qpcrit_1
    print(f"AR(phat): {reject_phat}")
    print(f"AR(1): {reject_1}")

    print("\np-values")
    print(f"AR(phat): {Qpval_phat:.4f}")
    print(f"AR(1): {Qpval_1:.4f}")

    # comment on results
    if reject_phat:
        print("\nResiduals from AR(phat) model show significant autocorrelation.")
    else:
        print("\nResiduals from AR(phat) model appear to be white noise.")

    if reject_1:
        print("Residuals from AR(1) model show significant autocorrelation.")
    else:
        print("Residuals from AR(1) model appear to be white noise.")

# example usage
if __name__ == "__main__":
    # load GNP deflator data with semicolon delimiter; Change path if necessary
    filepath = '/Users/aaronfeldman/Desktop/Quant_macro/Quantitative-Macroeconomics-2024-week-3/data/gnpdeflator.csv'
    gnp_data = pd.read_csv(filepath, delimiter=';')

    # display structure of data for debugging
    print("First few rows of the dataset:")
    print(gnp_data.head())
    print("\nColumn names:")
    print(gnp_data.columns)

    # update the column selection based on the dataset
    try:
        price_index = gnp_data['gnpdeflator'].values  # column is named 'gnpdeflator'
    except KeyError:
        raise KeyError("The dataset does not contain a 'gnpdeflator' column. Please check the structure of your data.")

    # compute the inflation series
    infl = np.diff(np.log(price_index))

    # run portmanteau test
    portmanteau_test(infl)

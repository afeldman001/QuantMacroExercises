import numpy as np
from scipy.stats import t
from ARp_OLS_w4 import ARpOLS  # Import the ARpOLS function

# function to simulate AR(4) data
def simulate_ar4(n, coefficients, constant, eps_std):
    """
    Simulates an AR(4) process with a constant.
    
    Parameters:
        n (int): Number of data points.
        coefficients (list): List of 4 coefficients for AR(4) model.
        constant (float): Constant term in AR(4) model.
        eps_std (float): Standard deviation of white noise.
        
    Returns:
        np.ndarray: Simulated AR(4) time series.
    """
    y = np.zeros(n)
    eps = np.random.normal(0, eps_std, n)
    
    # generate AR(4) process
    for t in range(4, n):
        y[t] = (constant + 
                coefficients[0] * y[t-1] + 
                coefficients[1] * y[t-2] + 
                coefficients[2] * y[t-3] + 
                coefficients[3] * y[t-4] + 
                eps[t])
        
    return y

# main function to run simulation and estimation and print results
def main():
    # parameters for AR(4) simulation
    n = 1000000  # number of observations
    coefficients = [0.5, -0.2, 0.1, 0.3]  # AR(4) coefficients
    constant = 1.0  # constant
    eps_std = 0.5  # standard deviation of noise
    
    # simulate AR(4) data
    y = simulate_ar4(n, coefficients, constant, eps_std)
    
    # estimate AR(4) model on simulated data using ARpOLS
    p = 4  # number of lags for AR(4)
    const = 1  # include constant
    alpha = 0.05  # significance level
    
    # call ARpOLS for estimation
    results = ARpOLS(y, p, const, alpha)
    
    # print results
    print("OLS Estimates (thetahat):", results["thetahat"])
    print("Standard Errors (sd_thetahat):", results["sd_thetahat"])
    print("t-Statistics (tstat):", results["tstat"])
    print("p-Values (pvalues):", results["pvalues"])
    print("Residual Std Deviation (siguhat):", results["siguhat"])
    print("Confidence Intervals (theta_ci):", results["theta_ci"])
    print("Residuals (resid):", results["resid"])

# run main function if script is executed directly
if __name__ == "__main__":
    main()

import numpy as np
import matplotlib.pyplot as plt

def irf_plots(a_comp, impact_matrix, n_steps, cumsum_indicator=None, variable_names=None, shock_names=None, no_plot=False):
    """
    compute impulse response functions for svar model
    b0*y_t = b_1*y(-1) + b_2*y(-2) + ... + b_p*y(-p) + e_t with e_t ~ iid(0,i)
    given companion matrix a_comp of the reduced-form var and impact matrix inv(b0)
    
    parameters:
    -----------
    a_comp : ndarray
        [(nvars*(nlags-1)) x (nvars*nlags)] companion matrix of reduced-form var
    impact_matrix : ndarray
        [nvars x nvars] impact matrix of svar model, either inv(b0) or inv(b0)*sqrt(e[epsilon_t*epsilon_t'])
    n_steps : int
        number of steps for impulse response function
    cumsum_indicator : ndarray, optional
        [nvars x 1] boolean vector, 1 indicates for which variable to compute the cumulative sum in irfs
    variable_names : list, optional
        [nvars x 1] list of strings with variable names
    shock_names : list, optional
        [nvars x 1] list of strings with structural shock names
    no_plot : bool, optional
        if true, turn off displaying of plots (useful for bootstrapping)
        
    returns:
    --------
    irf_point : ndarray
        [nvars, nvars, n_steps+1] array containing the irf of the 'j' variable to the 'k' shock
    """
    n_vars = impact_matrix.shape[0]
    n_lag = a_comp.shape[1] // n_vars
    
    # set default options if not specified
    if cumsum_indicator is None:
        cumsum_indicator = np.zeros(n_vars, dtype=bool)
    if variable_names is None:
        variable_names = [f"y{i+1}" for i in range(n_vars)]
    if shock_names is None:
        shock_names = [f"e{i+1}" for i in range(n_vars)]
        
    # initialize variables
    irf_point = np.full((n_vars, n_vars, n_steps + 1), np.nan)
    j = np.hstack([np.eye(n_vars), np.zeros((n_vars, n_vars * (n_lag - 1)))])
    
    # compute the impulse response function using the companion matrix
    ah = np.eye(a_comp.shape[0])  # initialize a^h at h=0
    jt_b0_inv = j.T @ impact_matrix
    
    for h in range(n_steps + 1):
        irf_point[:, :, h] = j @ ah @ jt_b0_inv
        ah = ah @ a_comp  # a^h = a^(h-1)*a
        
    # use cumsum to get response of level variables from original variables in differences
    for ivar in range(n_vars):
        if cumsum_indicator[ivar]:
            irf_point[ivar, :, :] = np.cumsum(irf_point[ivar, :, :], axis=1)
            
    # plot
    if not no_plot:
        # define a timeline
        steps = np.arange(n_steps + 1)
        x_axis = np.zeros(n_steps + 1)
        
        plt.figure(figsize=(15, 10))
        count = 1
        
        for ivars in range(n_vars):  # index for variables
            for ishocks in range(n_vars):  # index for shocks
                irfs = irf_point[ivars, ishocks, :]
                plt.subplot(n_vars, n_vars, count)
                plt.plot(steps, irfs, linewidth=2)
                plt.plot(steps, x_axis, 'k', linewidth=2)
                plt.xlim([0, n_steps])
                plt.title(shock_names[ishocks], fontweight='bold', fontsize=10)
                plt.ylabel(variable_names[ivars], fontweight='bold', fontsize=10)
                plt.tick_params(labelsize=16)
                count += 1
                
        plt.tight_layout()
        plt.show()
        
    return irf_point


# implementation example demonstrating irf_plots function
np.random.seed(42)

# define model parameters
n_vars = 3  # GDP growth, inflation, interest rate
n_lags = 2
T = 1000    # number of time periods

# create VAR coefficients representing economic relationships
# first lag coefficients (A1)
a1 = np.array([
    [0.50,  0.10, -0.15],  # GDP growth equation
    [0.20,  0.40, -0.10],  # inflation equation
    [0.15,  0.30,  0.60]   # interest rate equation
])

# second lag coefficients (A2)
a2 = np.array([
    [0.20,  0.05, -0.05],
    [0.10,  0.15, -0.05],
    [0.05,  0.10,  0.20]
])

# create companion matrix
a_comp = np.zeros((n_vars * n_lags, n_vars * n_lags))
a_comp[0:n_vars, 0:n_vars] = a1
a_comp[0:n_vars, n_vars:2*n_vars] = a2
a_comp[n_vars:2*n_vars, 0:n_vars] = np.eye(n_vars)

# simulate data and compute impact matrix
y = np.zeros((T, n_vars))
residuals = np.random.normal(0, 1, (T, n_vars))

# generate VAR data
for t in range(2, T):
    y[t] = (a1 @ y[t-1].T + a2 @ y[t-2].T + residuals[t])

# Calculate residuals (U) from the VAR model
U = residuals[2:] - (y[1:-1] @ a1.T + y[:-2] @ a2.T)  # Get VAR residuals

# Calculate variance-covariance matrix using ML method
degrees_of_freedom = T - 2 - n_vars * n_lags  # Adjust for lags and parameters
sigma = (U.T @ U) / degrees_of_freedom

# Compute Cholesky decomposition for impact matrix
impact_matrix = np.linalg.cholesky(sigma)

# compute Cholesky decomposition for impact matrix
impact_matrix = np.linalg.cholesky(sigma)

# set parameters for IRF analysis
n_steps = 24  # show responses for 2 years (24 months)
cumsum_indicator = np.array([False, False, False])
variable_names = ['GDP Growth', 'Inflation', 'Interest Rate']
shock_names = ['Supply Shock', 'Demand Shock', 'Monetary Shock']

# calculate impulse responses
irf_result = irf_plots(
    a_comp=a_comp,
    impact_matrix=impact_matrix,
    n_steps=n_steps,
    cumsum_indicator=cumsum_indicator,
    variable_names=variable_names,
    shock_names=shock_names,
    no_plot=False
)

# display key results
print("\nImpulse Response Analysis Results:")
print("----------------------------------")
print(f"Model dimensions: {n_vars} variables, {n_lags} lags")
print(f"Forecast horizon: {n_steps} periods")

print("\nInitial Responses (Period 0):")
for i, var in enumerate(variable_names):
    for j, shock in enumerate(shock_names):
        print(f"{var} to {shock}: {irf_result[i, j, 0]:.3f}")
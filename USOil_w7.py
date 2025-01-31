# import required libraries for numerical operations, data handling, and visualization
import numpy as np
import pandas as pd
from scipy.linalg import cholesky
import matplotlib.pyplot as plt
import requests
from io import StringIO

class VARModel:
    def __init__(self, data, lags, const=True):
        """
        initialize vector autoregression model
        args:
            data: time series data matrix
            lags: number of lags to include
            const: boolean indicating whether to include a constant term
        """
        self.data = data
        self.lags = lags
        self.const = const
        self.n_vars = data.shape[1]  # number of variables in the system
        self.fit_model()
    
    def create_lags(self):
        """create matrix of lagged variables for VAR estimation"""
        # determine dimensions of the lag matrix
        nobs = len(self.data)
        n_lagged_terms = self.n_vars * self.lags + (1 if self.const else 0)
        ldata = np.zeros((nobs - self.lags, n_lagged_terms))
        
        # fill the lag matrix with appropriate values
        for i in range(self.lags):
            start_idx = i * self.n_vars
            end_idx = (i + 1) * self.n_vars
            ldata[:, start_idx:end_idx] = self.data[self.lags-i-1:-i-1]
        
        # add constant term if specified
        if self.const:
            ldata[:, -1] = 1
            
        return ldata
    
    def fit_model(self):
        """estimate VAR model parameters using OLS"""
        # prepare dependent and independent variables
        y = self.data[self.lags:]
        x = self.create_lags()
        
        # compute OLS estimates with numerical stability term
        xtx = x.T @ x + np.eye(x.shape[1]) * 1e-12
        self.beta = np.linalg.solve(xtx, x.T @ y)
        
        # compute residuals and variance-covariance matrix
        self.residuals = y - x @ self.beta
        self.sigma = (self.residuals.T @ self.residuals) / (len(self.residuals) - x.shape[1])
        
        # construct companion matrix for impulse response analysis
        comp_dim = self.n_vars * self.lags
        self.companion_matrix = np.zeros((comp_dim, comp_dim))
        beta_coeffs = self.beta[:-1].T if self.const else self.beta.T
        self.companion_matrix[0:self.n_vars, 0:self.n_vars*self.lags] = beta_coeffs
        
        # add identity matrices for higher-order dynamics
        if self.lags > 1:
            self.companion_matrix[self.n_vars:, :-self.n_vars] = np.eye(self.n_vars * (self.lags - 1))

def compute_irf(companion_matrix, B0inv, n_steps, n_vars):
    """
    compute structural impulse response functions
    args:
        companion_matrix: VAR companion matrix
        B0inv: structural impact matrix (Cholesky decomposition)
        n_steps: number of periods for IRF computation
        n_vars: number of variables in the system
    """
    # initialize impulse response matrix
    irf = np.zeros((n_steps + 1, n_vars, n_vars))
    irf[0, :, :] = B0inv
    
    # compute impulse responses for each period
    for i in range(1, n_steps + 1):
        temp = np.linalg.matrix_power(companion_matrix, i)
        irf[i, :, :] = temp[0:n_vars, 0:n_vars] @ B0inv
    
    return irf

def plot_irf(irf, var_names, shock_names, cumsum_indicator):
    """
    create plots of impulse response functions
    args:
        irf: impulse response functions array
        var_names: names of variables
        shock_names: names of structural shocks
        cumsum_indicator: boolean list for cumulative responses
    """
    # setup the plot grid
    n_vars = len(var_names)
    fig, axes = plt.subplots(n_vars, n_vars, figsize=(12, 10))
    fig.subplots_adjust(hspace=0.4, wspace=0.3)
    
    # create individual subplot for each variable-shock combination
    for i in range(n_vars):
        for j in range(n_vars):
            ax = axes[i, j]
            # compute response, applying cumulative sum if indicated
            response = np.cumsum(irf[:, i, j]) if cumsum_indicator[i] else irf[:, i, j]
            
            # plot styling and formatting
            ax.plot(response, color='#2c3e50', linewidth=2)
            ax.axhline(y=0, color='#e74c3c', linestyle='--', alpha=0.5)
            ax.grid(True, color='#ecf0f1')
            ax.set_title(f'{var_names[i]} to {shock_names[j]}', fontsize=10)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # set axis limits for better visualization
            y_max = np.max(np.abs(response))
            ax.set_ylim(-y_max*1.1, y_max*1.1)
            ax.set_xlim(0, len(response)-1)
            
            # add axis labels where appropriate
            if i == n_vars-1:
                ax.set_xlabel('Periods', fontsize=9)
            if j == 0:
                ax.set_ylabel('Response', fontsize=9)
    
    fig.suptitle('Structural Impulse Response Functions', fontsize=12, y=0.95)
    return fig

def main():
    """main function to execute the SVAR analysis"""
    # fetch and prepare data
    url = "https://raw.githubusercontent.com/wmutschl/Quantitative-Macroeconomics/refs/heads/main/data/USOil.csv"
    data = pd.read_csv(url)[['drpoil', 'infl', 'drgdp']].values
    
    # estimate VAR model and compute Cholesky decomposition
    var_model = VARModel(data, lags=4, const=True)
    B0inv_chol = cholesky(var_model.sigma, lower=True)
    
    # setup parameters for IRF analysis
    var_names = ["Real Price of Oil", "GDP Deflator Inflation", "Real GDP"]
    shock_names = ["Oil Price Shock", "eps2 Shock", "eps3 Shock"]
    cumsum_indicator = [1, 0, 1]  # indicate which responses should be cumulative
    
    # compute and plot impulse responses
    irf_chol = compute_irf(var_model.companion_matrix, B0inv_chol, 30, var_model.n_vars)
    plot_irf(irf_chol, var_names, shock_names, cumsum_indicator)
    plt.show()

if __name__ == "__main__":
    main()
# import necessary libraries
import numpy as np  # for numerical computations and matrix operations
import pandas as pd  # for data loading and processing
import matplotlib.pyplot as plt  # for plotting data
from datetime import datetime  # for handling dates
from dateutil.relativedelta import relativedelta  # for date increments by quarters


# define a class for estimating a VAR model in reduced form using OLS
class VARReducedForm:
    def __init__(self, ENDO, nlag, opt=None):
        # initialize VAR model and set default options
        if opt is None:  # set default options if none provided
            opt = {"const": 1, "dispestim": True, "eqOLS": True}

        # store input data and options
        self.ENDO = ENDO  # matrix of endogenous variables
        self.nlag = nlag  # number of lags for VAR model
        self.opt = opt  # dictionary containing options

        # get number of observations and variables
        self.nobs, self.nvar = ENDO.shape
        self.nobs_eff = self.nobs - nlag  # effective number of observations after lags

        # check for invalid inputs
        if self.nlag < 1:
            raise ValueError("Number of lags must be positive.")
        if self.nobs < self.nvar:
            raise ValueError("Number of observations is smaller than the number of variables. Transpose ENDO.")

        # create dependent and regressor matrices
        self.Y = self.create_dependent_matrix()
        self.Z = self.create_regressor_matrix()

        # estimate coefficients and covariance matrices
        self.A, self.U, self.SigmaOLS, self.SigmaML = self.estimate_coefficients()

        # compute companion matrix and maximum eigenvalue for stability analysis
        self.Acomp, self.maxEig = self.compute_companion_matrix()

        # initialize equation-by-equation OLS results as an empty dictionary
        self.equation_OLS_results = {}
        if self.opt.get("eqOLS", False):  # estimate individual equations if eqOLS is True
            self.equation_OLS_results = self.estimate_equation_by_equation()

        # display results if dispestim is True
        if self.opt.get("dispestim", False):
            self.display_results()

    # create dependent matrix Y (shifted values of endogenous variables)
    def create_dependent_matrix(self):
        return self.ENDO[self.nlag:, :].T  # transpose for column-wise storage

    # create regressor matrix Z (lagged values of endogenous variables)
    def create_regressor_matrix(self):
        Z = []  # initialize list to store lagged values
        for lag in range(1, self.nlag + 1):  # iterate through lags
            Z.append(self.ENDO[self.nlag - lag:-(lag), :])  # lagged endogenous variables
        Z = np.hstack(Z).T  # horizontally stack and transpose

        # add deterministic terms (constant or trend) if specified in options
        if self.opt.get("const", 1) == 1:  # add constant
            Z = np.vstack([np.ones((1, self.nobs_eff)), Z])
        elif self.opt.get("const", 1) == 2:  # add constant and linear trend
            trend = np.arange(self.nlag + 1, self.nobs + 1).reshape(1, -1)
            Z = np.vstack([np.ones((1, self.nobs_eff)), trend, Z])

        return Z

    # estimate coefficients (A), residuals (U), and covariance matrices
    def estimate_coefficients(self):
        A = self.Y @ self.Z.T @ np.linalg.inv(self.Z @ self.Z.T)  # OLS estimate of coefficients
        U = self.Y - A @ self.Z  # compute residuals
        UUt = U @ U.T  # sum of squared residuals

        # calculate OLS and ML covariance matrices
        SigmaOLS = UUt / (self.nobs_eff - self.nvar * self.nlag - self.opt.get("const", 1))
        SigmaML = UUt / self.nobs_eff

        return A, U, SigmaOLS, SigmaML

    # compute companion matrix and its maximum eigenvalue
    def compute_companion_matrix(self):
        A1_to_p = self.A[:, self.opt.get("const", 1):]  # select lagged coefficients
        Acomp = np.zeros((self.nvar * self.nlag, self.nvar * self.nlag))  # initialize companion matrix
        Acomp[:self.nvar, :] = A1_to_p  # fill first block with lagged coefficients
        if self.nlag > 1:  # fill lower block for higher lags
            Acomp[self.nvar:, :-self.nvar] = np.eye(self.nvar * (self.nlag - 1))

        # calculate maximum absolute eigenvalue (used for stability check)
        maxEig = np.max(np.abs(np.linalg.eigvals(Acomp)))

        return Acomp, maxEig

    # estimate coefficients for each equation individually using OLS
    def estimate_equation_by_equation(self):
        results = {}  # initialize dictionary to store results
        for i in range(self.nvar):  # iterate over variables
            y = self.Y[i, :]  # dependent variable for equation i
            x = self.Z.T  # regressors

            # estimate coefficients using OLS
            beta = np.linalg.inv(x.T @ x) @ (x.T @ y)
            yhat = x @ beta  # predicted values
            resid = y - yhat  # residuals
            sige = np.sum(resid**2) / (len(y) - x.shape[1])  # variance of residuals

            # store results for equation i
            results[f"eq{i+1}"] = {
                "beta": beta,
                "yhat": yhat,
                "resid": resid,
                "sige": sige,
                "rsqr": 1 - np.sum(resid**2) / np.sum((y - np.mean(y))**2),
            }
        return results

    # display estimation results
    def display_results(self):
        print("Estimated Coefficients (A):\n", self.A)
        print("OLS Covariance Matrix (SigmaOLS):\n", self.SigmaOLS)
        print("ML Covariance Matrix (SigmaML):\n", self.SigmaML)
        print("Maximum Eigenvalue of Companion Matrix:", self.maxEig)


# Visualize and estimate 3-equation VAR(4) model with OLS
# -------------------------------------------------------------------------

# load data from CSV file
data = pd.read_csv('/Users/aaronfeldman/Desktop/Quant_macro/threeVariableVAR.csv')

# clean unnecessary columns: drop empty or unnamed columns
data = data.dropna(axis=1, how='all')
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

# verify cleaned data
print("Cleaned dataset shape:", data.shape)
print("Columns in dataset:", data.columns)

# convert cleaned data to numpy array for VAR estimation
ENDO = data.values

# define variable names for plotting and reporting
varnames = ['Real GNP Growth', 'Federal Funds Rate', 'GNP Deflator Inflation']

# dynamically adjust varnames if there are more variables than expected
if len(varnames) < ENDO.shape[1]:
    varnames += [f"Variable {i+1}" for i in range(len(varnames), ENDO.shape[1])]

# ensure time alignment with expected data size
if ENDO.shape[0] != 213:
    raise ValueError("Mismatch between dataset rows and expected time points.")

# initialize and estimate VAR(4) model
nlag = 4  # number of lags
opt = {"const": 1, "eqOLS": True}  # options for VAR model
VAR4 = VARReducedForm(ENDO, nlag, opt)

# print maximum eigenvalue for stability check
print("Maximum Eigenvalue:", VAR4.maxEig)

# print confidence intervals for estimated coefficients
for eq_name, eq_results in VAR4.equation_OLS_results.items():
    print(f"Confidence Intervals for {eq_name}:")
    beta = eq_results["beta"]  # regression coefficients
    std_err = np.sqrt(eq_results["sige"] * np.diag(np.linalg.inv(VAR4.Z @ VAR4.Z.T)))  # standard errors
    conf_int = np.vstack([beta - 1.96 * std_err, beta + 1.96 * std_err]).T  # 95% confidence intervals
    print(conf_int)

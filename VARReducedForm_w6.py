import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil.relativedelta import relativedelta


class VARReducedForm:
    def __init__(self, ENDO, nlag, opt=None):
        if opt is None:
            opt = {"const": 1, "dispestim": True, "eqOLS": True}

        self.ENDO = ENDO
        self.nlag = nlag
        self.opt = opt
        self.nobs, self.nvar = ENDO.shape
        self.nobs_eff = self.nobs - nlag

        if self.nlag < 1:
            raise ValueError("Number of lags must be positive.")
        if self.nobs < self.nvar:
            raise ValueError("Number of observations is smaller than the number of variables. Transpose ENDO.")

        self.Y = self.create_dependent_matrix()
        self.Z = self.create_regressor_matrix()
        self.A, self.U, self.SigmaOLS, self.SigmaML = self.estimate_coefficients()
        self.Acomp, self.maxEig = self.compute_companion_matrix()

        # Always initialize equation_OLS_results
        self.equation_OLS_results = {}
        if self.opt.get("eqOLS", False):
            self.equation_OLS_results = self.estimate_equation_by_equation()

        if self.opt.get("dispestim", False):
            self.display_results()

    def create_dependent_matrix(self):
        return self.ENDO[self.nlag:, :].T

    def create_regressor_matrix(self):
        Z = []
        for lag in range(1, self.nlag + 1):
            Z.append(self.ENDO[self.nlag - lag:-(lag), :])
        Z = np.hstack(Z).T

        if self.opt.get("const", 1) == 1:
            Z = np.vstack([np.ones((1, self.nobs_eff)), Z])
        elif self.opt.get("const", 1) == 2:
            trend = np.arange(self.nlag + 1, self.nobs + 1).reshape(1, -1)
            Z = np.vstack([np.ones((1, self.nobs_eff)), trend, Z])

        return Z

    def estimate_coefficients(self):
        A = self.Y @ self.Z.T @ np.linalg.inv(self.Z @ self.Z.T)
        U = self.Y - A @ self.Z
        UUt = U @ U.T

        SigmaOLS = UUt / (self.nobs_eff - self.nvar * self.nlag - self.opt.get("const", 1))
        SigmaML = UUt / self.nobs_eff

        return A, U, SigmaOLS, SigmaML

    def compute_companion_matrix(self):
        A1_to_p = self.A[:, self.opt.get("const", 1):]
        Acomp = np.zeros((self.nvar * self.nlag, self.nvar * self.nlag))
        Acomp[:self.nvar, :] = A1_to_p
        if self.nlag > 1:
            Acomp[self.nvar:, :-self.nvar] = np.eye(self.nvar * (self.nlag - 1))

        maxEig = np.max(np.abs(np.linalg.eigvals(Acomp)))

        return Acomp, maxEig

    def estimate_equation_by_equation(self):
        results = {}
        for i in range(self.nvar):
            y = self.Y[i, :]
            x = self.Z.T

            beta = np.linalg.inv(x.T @ x) @ (x.T @ y)
            yhat = x @ beta
            resid = y - yhat
            sige = np.sum(resid**2) / (len(y) - x.shape[1])

            results[f"eq{i+1}"] = {
                "beta": beta,
                "yhat": yhat,
                "resid": resid,
                "sige": sige,
                "rsqr": 1 - np.sum(resid**2) / np.sum((y - np.mean(y))**2),
            }
        return results

    def display_results(self):
        print("Estimated Coefficients (A):\n", self.A)
        print("OLS Covariance Matrix (SigmaOLS):\n", self.SigmaOLS)
        print("ML Covariance Matrix (SigmaML):\n", self.SigmaML)
        print("Maximum Eigenvalue of Companion Matrix:", self.maxEig)


# Main Script
# -------------------------------------------------------------------------

# Load data and clean unnecessary columns
data = pd.read_csv('/Users/aaronfeldman/Desktop/Quant_macro/threeVariableVAR.csv')

# Remove blank or unnamed columns
data = data.dropna(axis=1, how='all')
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

# Verify the cleaned data
print("Cleaned dataset shape:", data.shape)
print("Columns in dataset:", data.columns)

# Convert cleaned data to matrix for VAR model
ENDO = data.values
varnames = ['Real GNP Growth', 'Federal Funds Rate', 'GNP Deflator Inflation']

# Dynamically adjust varnames
if len(varnames) < ENDO.shape[1]:
    varnames += [f"Variable {i+1}" for i in range(len(varnames), ENDO.shape[1])]

# Ensure time alignment
if ENDO.shape[0] != 213:
    raise ValueError("Mismatch between dataset rows and expected time points.")

# VAR(4) estimation with OLS
nlag = 4
opt = {"const": 1, "eqOLS": True}
VAR4 = VARReducedForm(ENDO, nlag, opt)

# Print maximum eigenvalue
print("Maximum Eigenvalue:", VAR4.maxEig)

# Confidence intervals for coefficients
for eq_name, eq_results in VAR4.equation_OLS_results.items():
    print(f"Confidence Intervals for {eq_name}:")
    beta = eq_results["beta"]
    std_err = np.sqrt(eq_results["sige"] * np.diag(np.linalg.inv(VAR4.Z @ VAR4.Z.T)))
    conf_int = np.vstack([beta - 1.96 * std_err, beta + 1.96 * std_err]).T
    print(conf_int)

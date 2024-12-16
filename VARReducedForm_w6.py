import numpy as np
from scipy.linalg import svd, eig

class VARReducedForm:
    def __init__(self, ENDO, nlag, opt=None):
        if nlag < 1:
            raise ValueError("nlag needs to be positive")
        
        self.ENDO = ENDO
        self.nlag = nlag
        self.opt = self._set_defaults(opt)

        self.nobs, self.nvar = ENDO.shape
        if self.nobs < self.nvar:
            raise ValueError("The number of observations is smaller than the number of variables; check 'ENDO' input.")
        
        self.nobs_eff = self.nobs - nlag

        self.Y, self.Z = self._prepare_data()
        self.A, self.U, self.SigmaOLS, self.SigmaML, self.Acomp, self.maxEig = self._estimate_model()
        self.eq_results = self._equation_by_equation_ols() if self.opt['eqOLS'] else None

    def _set_defaults(self, opt):
        if opt is None:
            opt = {}
        opt.setdefault('const', 1)
        if opt['const'] not in [0, 1, 2]:
            raise ValueError("'opt.const' can only take values 0, 1, or 2")
        opt.setdefault('dispestim', True)
        opt.setdefault('eqOLS', True)
        return opt

    def _prepare_data(self):
        Y = self.ENDO[self.nlag:].T
        Z = self._lag_matrix(self.ENDO, self.nlag).T

        # adjust Z to have nobs_eff columns
        Z = Z[:, :self.nobs_eff]

        if self.opt['const'] == 1:
            Z = np.vstack([np.ones((1, self.nobs_eff)), Z])
        elif self.opt['const'] == 2:
            trend = np.arange(self.nlag + 1, self.nobs + 1).reshape(1, -1)
            Z = np.vstack([np.ones((1, self.nobs_eff)), trend, Z])

        return Y, Z

    def _lag_matrix(self, data, lags):
        nobs, nvar = data.shape
        lagged_data = []

        for lag in range(1, lags + 1):
            lagged = np.vstack([np.full((lag, nvar), np.nan), data[:-lag]])
            lagged_data.append(lagged)

        lagged_matrix = np.hstack(lagged_data)
        return lagged_matrix[lags:]  # remove initial rows with NaNs

    def _estimate_model(self):
        A = self.Y @ self.Z.T @ np.linalg.inv(self.Z @ self.Z.T)
        U = self.Y - A @ self.Z
        UUt = U @ U.T

        SigmaOLS = UUt / (self.nobs_eff - self.nvar * self.nlag - self.opt['const'])
        SigmaML = UUt / self.nobs_eff

        Acomp = np.vstack([
            np.hstack([A[:, self.opt['const']:(self.nvar * self.nlag + self.opt['const'])], np.zeros((self.nvar, self.nvar * (self.nlag - 1)))]),
            np.hstack([np.eye(self.nvar * (self.nlag - 1)), np.zeros((self.nvar * (self.nlag - 1), self.nvar))])
        ])
        maxEig = np.max(np.abs(eig(Acomp)[0]))

        return A, U, SigmaOLS, SigmaML, Acomp, maxEig

    def _equation_by_equation_ols(self):
        results = {}
        for j in range(self.nvar):
            y = self.Y[j, :]
            x = self.Z.T
            results[f'eq{j+1}'] = self._ols_model(y, x)
        return results

    def _ols_model(self, y, x):
        T, K = x.shape
        xtx_inv = np.linalg.inv(x.T @ x)
        beta = xtx_inv @ (x.T @ y)
        yhat = x @ beta
        resid = y - yhat

        sigu = resid.T @ resid
        sige = sigu / (T - K)

        sigb = np.sqrt(np.diag(sige * xtx_inv))
        tstat = beta / sigb

        rsqr = 1 - sigu / np.sum((y - np.mean(y)) ** 2)
        rbar = 1 - ((1 - rsqr) * (T - 1) / (T - K - 1))

        return {
            'beta': beta,
            'yhat': yhat,
            'resid': resid,
            'sige': sige,
            'bstd': sigb,
            'tstat': tstat,
            'rsqr': rsqr,
            'rbar': rbar
        }

    def display_results(self):
        if not self.opt['dispestim']:
            return

        if self.opt['const'] == 1:
            print("Constant coefficients:", self.A[:, 0])
        elif self.opt['const'] == 2:
            print("Constant and trend coefficients:", self.A[:, :2])

        print("Sigma OLS:", self.SigmaOLS)
        print("Sigma ML:", self.SigmaML)
        print("Max Eigenvalue of Companion Matrix:", self.maxEig)


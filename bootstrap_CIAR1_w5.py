import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from ARp_OLS_w4 import ARpOLS  # import ARpOLS function

# set options
B = 10000  # number of bootstrap repetitions
T = 100  # sample size
burnin = 100  # number of observations to discard
alpha = 0.05  # confidence level

# generate true data from AR(1)
u = np.random.exponential(scale=1.0, size=T + burnin) - 1  # draw from exponential and subtract mean
y = np.empty(T + burnin) # initialize data vector
c = 1  # AR(1) constant
phi = 0.8  # AR(1) coefficient
y[0] = c / (1 - phi)  # initialize with the mean

for t in range(1, T + burnin):
    y[t] = c + phi * y[t - 1] + u[t]  # generate AR(1) process

y = y[burnin:]  # discard burn-in observations

# OLS estimation and t-statistic on true data
ols_results = ARpOLS(y, 1, 1, alpha)
uhat = ols_results["resid"]
chat = ols_results["thetahat"][0]
phihat = ols_results["thetahat"][1]
sig_phihat = ols_results["sd_thetahat"][1]
sig_uhat = ols_results["siguhat"]

# Percentile-t Bootstrap
taustar_parametric = np.empty(B) # initialize t statistics output vector
taustar_non_parametric = np.empty(B) # initialize t statistics output vector

for b in range(B):
    # draw with replacement
    ustar_parametric = np.random.normal(0, sig_uhat, len(uhat))
    ustar_non_parametric = np.random.choice(uhat, size=len(uhat), replace=True)
    
    # initialize artificial data vectors
    ystar_parametric = np.empty_like(y)
    ystar_non_parametric = np.empty_like(y)
    
    # initialize first observation with real data
    ystar_parametric[0] = y[0]
    ystar_non_parametric[0] = y[0]
    
    for t in range(1, len(y)):
        # generate artificial data from AR(1)
        ystar_parametric[t] = chat + phihat * ystar_parametric[t - 1] + ustar_parametric[t - 1]
        ystar_non_parametric[t] = chat + phihat * ystar_non_parametric[t - 1] + ustar_non_parametric[t - 1]
    
    # OLS estimation and t-statistic on artificial data
    olsstar_parametric = ARpOLS(ystar_parametric, 1, 1, alpha)
    olsstar_non_parametric = ARpOLS(ystar_non_parametric, 1, 1, alpha)
    
    phistar_parametric = olsstar_parametric["thetahat"][1]
    phistar_non_parametric = olsstar_non_parametric["thetahat"][1]
    sig_phistar_parametric = olsstar_parametric["sd_thetahat"][1]
    sig_phistar_non_parametric = olsstar_non_parametric["sd_thetahat"][1]
    
    taustar_parametric[b] = (phistar_parametric - phihat) / sig_phistar_parametric
    taustar_non_parametric[b] = (phistar_non_parametric - phihat) / sig_phistar_non_parametric

# bootstrap distribution
taustar_parametric.sort() # sort output vector to access quantiles
taustar_non_parametric.sort() # sort output vector to access quantiles

# bootstrap confidence intervals
lower_boot_parametric = phihat - taustar_parametric[int((1 - alpha / 2) * B)] * sig_phihat # lower bound for bootstrap CI (parametric)
upper_boot_parametric = phihat - taustar_parametric[int(alpha / 2 * B)] * sig_phihat # upper bound for bootstrap CI (parametric)
lower_boot_non_parametric = phihat - taustar_non_parametric[int((1 - alpha / 2) * B)] * sig_phihat # lower bound for bootstrap CI (non-parametric)
upper_boot_non_parametric = phihat - taustar_non_parametric[int(alpha / 2 * B)] * sig_phihat # upper bound for bootstrap CI (non-parametric)

# asymptotic distribution
z = norm.ppf(1 - alpha / 2) # quantile function (percent point function)
lower_approx = phihat - z * sig_phihat
upper_approx = phihat + z * sig_phihat

# print results
print("\nConfidence Intervals:")
print(f"Asymptotic CI:               [{lower_approx:.4f}, {upper_approx:.4f}]")
print(f"Parametric Bootstrap CI:     [{lower_boot_parametric:.4f}, {upper_boot_parametric:.4f}]")
print(f"Non-Parametric Bootstrap CI: [{lower_boot_non_parametric:.4f}, {upper_boot_non_parametric:.4f}]\n")

# plot bootstrap distributions
x = np.linspace(-5, 5, 500)
plt.figure(figsize=(12, 6))
plt.suptitle("Bootstrap Distributions of taustar")

plt.subplot(1, 2, 1)
plt.hist(taustar_parametric, bins=50, density=True, alpha=0.7, label="Parametric")
plt.plot(x, norm.pdf(x, 0, 1), label="Standard Normal")
plt.title("Parametric Bootstrap")
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(taustar_non_parametric, bins=50, density=True, alpha=0.7, label="Non-Parametric")
plt.plot(x, norm.pdf(x, 0, 1), label="Standard Normal")
plt.title("Non-Parametric Bootstrap")
plt.legend()

plt.tight_layout()
plt.show()

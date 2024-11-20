import numpy as np
import matplotlib.pyplot as plt

# initialize parameter vector x = [phi, sigma, T]
phi_values = [1, 0.7, 0.6, 0.35, 0.1, 0.03]  # vector of AR(1) coefficients 
sigma = 1                                    # standard deviation of the noise
T = 2000                                     # number of observations  
max_lags = 30                                # number of lags to calculate ACF for

# define function to compute empirical autocorrelation
def emp_acf(y, lag):
    length = len(y)
    mean = np.mean(y)
    cov = np.sum((y[:length - lag] - mean) * (y[lag:] - mean)) / (length - lag)
    var = np.sum((y - mean) ** 2) / length
    return cov / var

# plot setup
plt.figure(figsize=(15, 12))

# iterate over different phi values
for i, phi in enumerate(phi_values, start=1):
    # simulate AR(1) process for the current phi
    np.random.seed(0)
    time_series = [0]  # Initial observation
    for t in range(1, T):
        time_series.append(phi * time_series[-1] + np.random.normal(0, sigma))
    time_series = np.array(time_series)

    # calculate ACF for the current time series
    acf_values = [emp_acf(time_series, lag) for lag in range(max_lags)]

    # plot time series
    plt.subplot(len(phi_values), 2, 2 * i - 1)
    plt.plot(time_series, color='blue')
    plt.title(f'Simulated AR(1) Time Series (phi={phi})')
    plt.xlabel('Time')
    plt.ylabel('Value')

    # plot empirical ACF manually
    plt.subplot(len(phi_values), 2, 2 * i)
    lags = range(max_lags)
    plt.vlines(lags, 0, acf_values, colors='tab:blue', lw=2)  # vertical lines
    plt.plot(lags, acf_values, 'o', color='tab:red')          # marker points
    plt.title(f'Empirical ACF (phi={phi})')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')

plt.tight_layout()
plt.show()

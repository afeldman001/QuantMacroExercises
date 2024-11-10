import numpy as np
import matplotlib.pyplot as plt

# Initialize parameter vector x = [phi, sigma, T]
phi_values = [1, 0.7, 0.6, 0.35, 0.1, 0.03]  # Vector of AR(1) coefficients 
sigma = 1                                    # Standard deviation of the noise
T = 2000                                     # Number of observations  
max_lags = 30                                # Number of lags to calculate ACF for

# Define function to compute empirical autocorrelation
def emp_acf(y, lag):
    length = len(y)
    mean = np.mean(y)
    cov = np.sum((y[:length - lag] - mean) * (y[lag:] - mean)) / (length - lag)
    var = np.sum((y - mean) ** 2) / length
    return cov / var

# Plot setup
plt.figure(figsize=(15, 12))

# Iterate over different phi values
for i, phi in enumerate(phi_values, start=1):
    # Simulate AR(1) process for the current phi
    np.random.seed(0)
    time_series = [0]  # Initial observation
    for t in range(1, T):
        time_series.append(phi * time_series[-1] + np.random.normal(0, sigma))
    time_series = np.array(time_series)

    # Calculate ACF for the current time series
    acf_values = [emp_acf(time_series, lag) for lag in range(max_lags)]

    # Plotting the time series
    plt.subplot(len(phi_values), 2, 2 * i - 1)
    plt.plot(time_series, color='blue')
    plt.title(f'Simulated AR(1) Time Series (phi={phi})')
    plt.xlabel('Time')
    plt.ylabel('Value')

    # Plotting the empirical ACF manually
    plt.subplot(len(phi_values), 2, 2 * i)
    lags = range(max_lags)
    plt.vlines(lags, 0, acf_values, colors='tab:blue', lw=2)  # Vertical lines
    plt.plot(lags, acf_values, 'o', color='tab:red')          # Marker points
    plt.title(f'Empirical ACF (phi={phi})')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')

plt.tight_layout()
plt.show()

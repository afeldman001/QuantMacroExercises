import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import geom, gamma, t

# Parameters
T = 10000  # Maximum horizon of periods
phi = 0.8  # AR(1) coefficient for the stable process

# Distribution parameters
sig_z, mu_z = 0.2, 10  # Normal distribution parameters
a, b = 2, 4  # Uniform distribution range
p = 0.2  # Geometric distribution probability
k, thet = 2, 2  # Gamma distribution parameters
nu1 = 8  # Degrees of freedom for Student's t with finite variance
nu2 = 2  # Degrees of freedom for Student's t with infinite variance

# Initialize arrays for storing results
Y = np.zeros((T, 6))  # Stores AR(1) series for each distribution

# Generate demeaned error terms for each distribution
error_distributions = {
    'Normal': mu_z + sig_z * np.random.randn(T),                  # Normal
    'Uniform': a + (b - a) * np.random.rand(T) - (a + b) / 2,     # Uniform, demeaned
    'Geometric': geom.rvs(p, size=T) - (1 / p),                   # Geometric, demeaned
    'Gamma': gamma.rvs(k, scale=thet, size=T) - (k * thet),       # Gamma, demeaned
    "Student's t (finite variance)": t.rvs(nu1, size=T),          # Student's t (finite variance)
    "Student's t (infinite variance)": t.rvs(nu2, size=T)         # Student's t (infinite variance)
}

# Labels and theoretical means for plotting
distribution_labels = list(error_distributions.keys())
theoretical_means = [0] * len(distribution_labels)  # All means are centered at 0

# Set up subplots for each distribution
fig, axes = plt.subplots(3, 2, figsize=(15, 12))
axes = axes.flatten()  # Flatten to easily iterate over

# Iterate over each distribution to compute and plot the means over growing sample sizes
for idx, (label, error_term) in enumerate(error_distributions.items()):
    means = []  # List to store mean for each growing sample size

    # Initialize the AR(1) process
    data = np.zeros(T)
    for t in range(1, T):
        data[t] = phi * data[t - 1] + error_term[t]  # AR(1) process with error terms

    # Calculate sample means for growing sample sizes
    for t in range(1, T + 1):
        sample_mean = np.mean(data[:t])  # Compute mean up to sample size t
        means.append(sample_mean)

    # Plot the sample means for this distribution
    ax = axes[idx]
    ax.plot(range(1, T + 1), means, label=f'Sample Mean ({label})', color='blue')
    ax.axhline(theoretical_means[idx], color='gray', linestyle='--', linewidth=0.7, label='Theoretical Mean')
    ax.set_xlabel('Sample Size')
    ax.set_ylabel('Sample Mean')
    ax.set_title(label)
    ax.legend()
    ax.grid(True, which="both", ls="--", linewidth=0.5)

# Adjust layout and show plot
plt.tight_layout()
plt.suptitle('Illustration of the Weak Law of Large Numbers for AR(1) Process with Different Error Term Distributions', y=1.02)
plt.show()


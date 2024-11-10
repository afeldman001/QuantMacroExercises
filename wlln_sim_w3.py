import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import geom, gamma, t

# Maximum sample size for illustrating the Weak Law of Large Numbers
T = 10000  # Maximum horizon of periods for smoother convergence illustration

# Distribution parameters (matching the MATLAB initialization)
sig_z, mu_z = 0.2, 10      # Parameters for normal distribution
a, b = 2, 4                # Parameters for uniform distribution (range [a, b])
p = 0.2                    # Parameter for geometric distribution (probability of success)
k, thet = 2, 2             # Parameters for gamma distribution
nu1 = 8                    # Degrees of freedom for Student's t with finite variance
nu2 = 2                    # Degrees of freedom for Student's t with infinite variance

# Theoretical means for each distribution (centered where necessary)
theoretical_means = {
    'Normal': mu_z,
    'Uniform': (a + b) / 2,
    'Geometric': (1 - p) / p,
    "Student's t (finite variance)": 0,
    "Student's t (infinite variance)": 0,
    'Gamma': k * thet
}

# Define the distributions with specified parameters
distributions = {
    'Normal': lambda size: mu_z + sig_z * np.random.randn(size),
    'Uniform': lambda size: a + (b - a) * np.random.rand(size),
    'Geometric': lambda size: geom.rvs(p, size=size) - (1 / p),  # Centered at theoretical mean
    "Student's t (finite variance)": lambda size: t.rvs(nu1, size=size),
    "Student's t (infinite variance)": lambda size: t.rvs(nu2, size=size),
    'Gamma': lambda size: gamma.rvs(k, scale=thet, size=size) - (k * thet)  # Centered at theoretical mean
}

# Set up subplots in a 3x2 grid for each distribution
fig, axes = plt.subplots(3, 2, figsize=(15, 12))
axes = axes.flatten()  # Flatten to easily iterate over

# Iterate over distributions and plot each in its own subplot
for ax, (dist_name, dist_func) in zip(axes, distributions.items()):
    means = []  # List to store the means for each sample size

    # Generate a single large sample (of size T) for each distribution
    data = dist_func(T)

    # Calculate sample means for every sample size from 1 to T
    for t in range(1, T + 1):
        sample_mean = np.mean(data[:t])  # Calculate the mean of the first t samples
        means.append(sample_mean)  # Store the mean for the current sample size

    # Plot results for the current distribution
    ax.plot(range(1, T + 1), means, label=f'Sample Mean ({dist_name})', color='blue')
    ax.axhline(theoretical_means[dist_name], color='gray', linestyle='--', linewidth=0.7, label='Theoretical Mean')
    ax.set_xlabel('Sample Size')
    ax.set_ylabel('Sample Mean')
    ax.set_title(dist_name)
    ax.legend()
    ax.grid(True, which="both", ls="--", linewidth=0.5)

# Adjust layout and show plot
plt.tight_layout()
plt.suptitle('Illustration of the Weak Law of Large Numbers for Different Distributions', y=1.02)
plt.show()

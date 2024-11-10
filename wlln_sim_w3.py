import numpy as np
import matplotlib.pyplot as plt

# Maximum sample size for illustrating the Weak Law of Large Numbers
T = 10000  # Set a manageable maximum sample size for smoother convergence illustration
num_trials = 100  # Number of trials for averaging (optional, helps reduce noise)

# Distributions with simplified names
distributions = {
    'Normal': lambda size: np.random.normal(0, 1, size),
    'Uniform': lambda size: np.random.uniform(0, 1, size),
    'Geometric': lambda size: np.random.geometric(0.5, size),
    "Student's t (finite variance)": lambda size: np.random.standard_t(2, size),
    "Student's t (infinite variance)": lambda size: np.random.standard_t(1, size),
    'Gamma': lambda size: np.random.gamma(2, 2, size)
}

# Theoretical means for each distribution
theoretical_means = {
    'Normal': 0,
    'Uniform': 0.5,
    'Geometric': 2,
    "Student's t (finite variance)": 0,
    "Student's t (infinite variance)": 0,
    'Gamma': 4
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

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Parameters for the AR(1) process
phi = 0.8  # AR(1) coefficient for dependence
mu, sigma = 0, 1  # Mean and standard deviation of Gaussian noise
sample_sizes = [50, 100, 500, 1000, 10000]  # Different sample sizes to test
num_trials = 1000  # Number of trials for each sample size

# Set up a 1x5 grid of subplots for each sample size
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
fig.suptitle("Central Limit Theorem for Dependent Data (Gaussian AR(1) Process)", y=1.05)

# Iterate over each sample size and simulate the sample means
for idx, n in enumerate(sample_sizes):
    means = []  # Store the sample means for each trial

    # Run multiple trials to collect sample means
    for _ in range(num_trials):
        # Initialize AR(1) process
        data = np.zeros(n)
        data[0] = np.random.normal(mu, sigma)  # Initial value with Gaussian noise

        # Generate AR(1) series for the current sample size
        for t in range(1, n):
            data[t] = phi * data[t - 1] + np.random.normal(mu, sigma)

        # Compute the sample mean and store it
        means.append(np.mean(data))

    # Plot the histogram of sample means for this sample size
    sns.histplot(means, kde=True, ax=axes[idx], color='skyblue')
    axes[idx].set_title(f'Sample Size = {n}')
    axes[idx].axvline(0, color='gray', linestyle='--')  # Expected mean line
    axes[idx].set_xlabel('Sample Mean')
    axes[idx].set_ylabel('Frequency')

# Adjust layout and display plot
plt.tight_layout()
plt.show()

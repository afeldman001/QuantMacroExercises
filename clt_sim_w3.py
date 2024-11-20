import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# parameters for AR(1) process
phi = 0.8  # AR(1) coefficient for dependence
mu, sigma = 0, 1  # mean and standard deviation of Gaussian noise
sample_sizes = [50, 100, 500, 1000, 10000]  # different sample sizes to test
num_trials = 1000  # number of trials for each sample size

# set up 1x5 grid of subplots for each sample size
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
fig.suptitle("Central Limit Theorem for Dependent Data (Gaussian AR(1) Process)", y=1.05)

# iterate over each sample size and simulate sample means
for idx, n in enumerate(sample_sizes):
    means = []  # store sample means for each trial

    # run multiple trials to collect sample means
    for _ in range(num_trials):
        # initialize AR(1) process
        data = np.zeros(n)
        data[0] = np.random.normal(mu, sigma)  # initial value with Gaussian noise

        # generate AR(1) series for current sample size
        for t in range(1, n):
            data[t] = phi * data[t - 1] + np.random.normal(mu, sigma)

        # compute sample mean and store
        means.append(np.mean(data))

    # plot histogram of sample means for this sample size
    sns.histplot(means, kde=True, ax=axes[idx], color='skyblue')
    axes[idx].set_title(f'Sample Size = {n}')
    axes[idx].axvline(0, color='gray', linestyle='--')  # Expected mean line
    axes[idx].set_xlabel('Sample Mean')
    axes[idx].set_ylabel('Frequency')

# adjust layout and display plot
plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import geom, gamma, t

# parameters
T = 10000  # maximum horizon of periods
phi = 0.8  # AR(1) coefficient for the stable process

# distribution parameters
sig_z, mu_z = 0.2, 10  # normal distribution parameters
a, b = 2, 4  # uniform distribution range
p = 0.2  # geometric distribution probability
k, thet = 2, 2  # gamma distribution parameters
nu1 = 8  # degrees of freedom for Student's t with finite variance
nu2 = 2  # degrees of freedom for Student's t with infinite variance

# initialize arrays for storing results
Y = np.zeros((T, 6))  # stores AR(1) series for each distribution

# generate demeaned error terms for each distribution
error_distributions = {
    'Normal': mu_z + sig_z * np.random.randn(T),                  # normal
    'Uniform': a + (b - a) * np.random.rand(T) - (a + b) / 2,     # uniform, demeaned
    'Geometric': geom.rvs(p, size=T) - (1 / p),                   # geometric, demeaned
    'Gamma': gamma.rvs(k, scale=thet, size=T) - (k * thet),       # gamma, demeaned
    "Student's t (finite variance)": t.rvs(nu1, size=T),          # student's t (finite variance)
    "Student's t (infinite variance)": t.rvs(nu2, size=T)         # student's t (infinite variance)
}

# labels and theoretical means for plotting
distribution_labels = list(error_distributions.keys())
theoretical_means = [0] * len(distribution_labels)  # all means centered at 0

# set up subplots for each distribution
fig, axes = plt.subplots(3, 2, figsize=(15, 12))
axes = axes.flatten()  # flatten to easily iterate over

# iterate over each distribution to compute and plot means over growing sample sizes
for idx, (label, error_term) in enumerate(error_distributions.items()):
    means = []  # list to store mean for each growing sample size

    # initialize the AR(1) process
    data = np.zeros(T)
    for t in range(1, T):
        data[t] = phi * data[t - 1] + error_term[t]  # AR(1) process with error terms

    # calculate sample means for growing sample sizes
    for t in range(1, T + 1):
        sample_mean = np.mean(data[:t])  # compute mean up to sample size t
        means.append(sample_mean)

    # plot the sample means for this distribution
    ax = axes[idx]
    ax.plot(range(1, T + 1), means, label=f'Sample Mean ({label})', color='blue')
    ax.axhline(theoretical_means[idx], color='gray', linestyle='--', linewidth=0.7, label='Theoretical Mean')
    ax.set_xlabel('Sample Size')
    ax.set_ylabel('Sample Mean')
    ax.set_title(label)
    ax.legend()
    ax.grid(True, which="both", ls="--", linewidth=0.5)

# adjust layout and show plot
plt.tight_layout()
plt.suptitle('Illustration of the Weak Law of Large Numbers for AR(1) Process with Different Error Term Distributions', y=1.02)
plt.show()


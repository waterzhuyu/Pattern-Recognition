import numpy as np
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
from matplotlib import pyplot as plt
import pandas as pd

"""
@Source: https://www.geeksforgeeks.org/non-parametric-density-estimation-methods-in-machine-learning/ 
sklearn reference for k neighbor and kernel method
"""

# Kernel Density Estimator using gaussian kernel
X = np.random.randn(100)
T = pd.Series(X)
T.hist()

plt.figure()
model = KernelDensity(kernel='gaussian',
                      bandwidth=0.2)
model.fit(X[:, None])
new_data = np.linspace(-5, 5, 1000)
density = np.exp(model.score_samples(new_data[:, None]))
# print(new_data)
# Plot the densities
plt.plot(new_data, density, '-',
         color='red')
plt.xlabel('Data')
plt.ylabel('Density')
plt.title('Kernel Density Estimator using Gaussian kernel')
plt.show()

# knn method
gaussian = norm(loc=0.5, scale=0.2)
X = gaussian.rvs(500)
grid = np.linspace(-0.1, 1.1, 1000)
k_set = [5, 10, 20]
fig, axes = plt.subplots(3, 1, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
    K = k_set[i]
    p = np.zeros_like(grid)
    n = X.shape[0]
    for i, x in enumerate(grid):
        dists = np.abs(X-x)
        neighbours = dists.argsort()
        neighbour_K = neighbours[K]
        p[i] = (K/n) * 1/(2 * dists[neighbour_K])
    ax.plot(grid, p, color='orange')
    ax.set_title(f'$k={K}$')
plt.show()

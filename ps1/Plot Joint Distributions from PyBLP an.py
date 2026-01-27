# Plot Joint Distributions from PyBLP and Our Code
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal

# Define parameters for Distribution 1
our_mean = [-1.54976786,  1.34234004]
our_cov = np.sqrt([[17.57353385, 20.5879952 ],
 [20.5879952,  27.43425161]])

# Define parameters for Distribution 2
pyblp_mean = [-1.949,  .84585]
pyblp_cov = np.sqrt([[19.754, 21.955 ],
 [21.955,  27.527]])

# Create grid
x = np.linspace(-10, 10, 1000)
y = np.linspace(-10, 10, 1000)
X, Y = np.meshgrid(x, y)

# Stack X and Y to create position array
pos = np.dstack((X, Y))

# Create the distributions
dist1 = multivariate_normal(our_mean, our_cov)
dist2 = multivariate_normal(pyblp_mean, pyblp_cov)

# Evaluate PDFs
Z1 = dist1.pdf(pos)
Z2 = dist2.pdf(pos)

# Create figure with single overlaid plot
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# Plot both distributions on same axes with transparency
surf1 = ax.plot_surface(X, Y, Z1, cmap='viridis', alpha=0.6, label='Our Distribution', antialiased=True)
surf2 = ax.plot_surface(X, Y, Z2, cmap='plasma', alpha=0.6, label='PyBLP Distribution', antialiased=True)

ax.set_xlabel('beta_price')
ax.set_ylabel('beta_x')
ax.set_zlabel('Probability Density')
ax.set_title('Overlaid Joint Distributions')

# Adjust viewing angle: elev=elevation (vertical), azim=azimuth (horizontal rotation)
ax.view_init(elev=20, azim=75)

# Add colorbars
cbar1 = fig.colorbar(surf1, ax=ax, shrink=0.5, pad=0.1)
cbar1.set_label('Our Distribution')
cbar2 = fig.colorbar(surf2, ax=ax, shrink=0.5, pad=0.15)
cbar2.set_label('PyBLP Distribution')

plt.tight_layout()
plt.show()
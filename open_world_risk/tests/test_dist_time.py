import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Generate arbitrary data with a bump (non-monotonic)
t = np.linspace(0, 10, 100)
# Create a function with a bump: start at 0, go up, then down, then up again
d = 2 * np.sin(t) + 0.5 * np.sin(3*t) + 3

# Calculate time intervals (dt) between samples
# Assume uniform sampling, so dt is constant
dt = t[1] - t[0]

# Create bins for distance
num_bins = 30
hist, bin_edges = np.histogram(d, bins=num_bins)

# For each bin, count how many samples fall in it, then multiply by dt to get total time spent in that bin
time_per_bin = hist * dt
# Bin centers for plotting
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

# KDE for PDF estimation
kde = gaussian_kde(d)
d_grid = np.linspace(d.min(), d.max(), 200)
pdf = kde(d_grid)

# Create figure with three subplots side by side
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Distance as a function of time
ax1.plot(t, d, 'b-', linewidth=2, label='Distance vs Time')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Distance (m)')
ax1.set_title('Distance as a Function of Time')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Plot 2: Dwell time at each distance (histogram)
ax2.bar(bin_centers, time_per_bin, width=(bin_edges[1]-bin_edges[0]), color='red', alpha=0.7, label='Time spent at distance')
ax2.set_xlabel('Distance (m)')
ax2.set_ylabel('Time spent (s)')
ax2.set_title('Time Spent at Each Distance')
ax2.grid(True, alpha=0.3)
ax2.legend()

# Plot 3: Probability density function (PDF) of distance
ax3.plot(d_grid, pdf, color='green', linewidth=2, label='PDF (KDE)')
ax3.set_xlabel('Distance (m)')
ax3.set_ylabel('Probability Density')
ax3.set_title('Probability Density Function of Distance')
ax3.grid(True, alpha=0.3)
ax3.legend()

# Adjust layout and display
plt.tight_layout()
plt.show()

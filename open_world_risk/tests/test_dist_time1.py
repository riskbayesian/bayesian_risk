import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.interpolate import make_interp_spline, BSpline

# Generate data with monotonically decreasing histogram and minimum distance of 0.5m
t = np.linspace(0, 10, 100)
# Create a function that starts at a higher distance and decreases over time
# This will give us a monotonically decreasing histogram
d = 1.2 + 1.0 * np.sin(0.7 * t) + 0.7 * np.cos(0.3 * t + 1.5) + 0.5
d = np.clip(d, 0.5, None)  # Ensure minimum distance is not violated

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
# Create grid from 0 to max distance to show the full range
d_grid = np.linspace(0, d.max(), 200)
pdf = kde(d_grid)

# Set PDF to 0 for distances below the minimum observed distance
min_distance = d.min()
pdf[d_grid < min_distance] = 0

# Create figure with four subplots in a 2x2 grid
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

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

# Plot 4:       Risk Field Prior (1 - PDF normalized as probability density)
prior_risk = 1 - pdf
# Set prior risk to 1 for distances below the minimum observed distance
prior_risk[d_grid < min_distance] = 1

# Find the first index after min_distance where prior_risk increases
start_idx = np.argmax(d_grid >= min_distance)
end_idx = None 
for i in range(start_idx, len(prior_risk) - 1):
    if prior_risk[i] < prior_risk[i + 1]:
        prior_risk[i + 1:] = 0
        end_idx = i+1
        break
    
#prior_risk = prior_risk[:end_idx]
#d_grid = d_grid[:end_idx]
prior_risk_0_to_1 = (prior_risk - np.amin(prior_risk)) / (np.amax(prior_risk) - np.amin(prior_risk))

ax4.plot(d_grid, prior_risk_0_to_1, color='orange', linewidth=2, label='Risk Field Prior')

# Find inflection points (changes in convexity) for control points
second_derivative = np.gradient(np.gradient(prior_risk_0_to_1, d_grid), d_grid)
inflection_indices = np.where(np.diff(np.sign(second_derivative)))[0]

# Always include endpoints
control_points_x = [d_grid[0]]
if len(inflection_indices) > 0:
    # Add inflection points
    control_points_x += list(d_grid[inflection_indices])
control_points_x.append(d_grid[-1])

# If too many, pick the most significant (by magnitude of second derivative)
num_control_points = 5
if len(control_points_x) > num_control_points:
    # Keep endpoints, pick most significant inflections
    endpoint_x = [d_grid[0], d_grid[-1]]
    inflection_x = d_grid[inflection_indices]
    inflection_strength = np.abs(second_derivative[inflection_indices])
    top_inflections = inflection_x[np.argsort(-inflection_strength)[:num_control_points-2]]
    control_points_x = [endpoint_x[0]] + sorted(top_inflections.tolist()) + [endpoint_x[1]]
elif len(control_points_x) < num_control_points:
    # If too few, fill in with evenly spaced points
    extra_needed = num_control_points - len(control_points_x)
    extra_x = np.linspace(d_grid[0], d_grid[-1], num_control_points)[1:-1]
    control_points_x = sorted(set(control_points_x + list(extra_x)))[0:num_control_points]

control_points_x = np.array(control_points_x)
control_points_y = np.interp(control_points_x, d_grid, prior_risk_0_to_1)
control_points = np.vstack([control_points_x, control_points_y]).T

# Construct knot vector for cubic B-spline
k = 3  # cubic
# Open uniform knot vector
knots = np.concatenate((np.repeat(control_points_x[0], k),
                        np.linspace(control_points_x[0], control_points_x[-1], num_control_points - k + 1),
                        np.repeat(control_points_x[-1], k)))
# Only y-values are used for 1D spline
spline = BSpline(knots, control_points_y, k)
d_grid_spline = np.linspace(d_grid[0], d_grid[-1], 300)
prior_risk_spline = spline(d_grid_spline)
ax4.plot(d_grid_spline, prior_risk_spline, color='blue', linestyle='--', linewidth=2, label='B-spline fit (4 ctrl pts)')
# Plot control points as red dots
ax4.plot(control_points_x, control_points_y, 'ro', markersize=8, label='Control points')

ax4.set_xlabel('Distance (m)')
ax4.set_ylabel('Risk')
ax4.set_title('Risk Field Prior (1 - PDF, Normalized: 0 to 1)')
ax4.grid(True, alpha=0.3)
ax4.legend()

# Adjust layout and display
plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline

# Define the problem parameters
degree = 3
n_control_points = 5
distance = 5

# Create linearly spaced knots
n_knots = n_control_points + degree + 1
knots = np.linspace(0, distance, n_knots - 2 * degree)
knots = np.concatenate(([0] * degree, knots, [distance] * degree))

# Define control points based on the conditions
# First control point at 1, last control point at 0 at distance 5
control_points = np.zeros((n_control_points, 2))
control_points[0] = [0, 1]  # First control point at (0, 1)
control_points[-1] = [distance, 0]  # Last control point at (5, 0)

# For the middle control points, we need them to sum to 1 and be monotonically decreasing
# We'll use a geometric sequence for the y-values
'''
middle_points = n_control_points - 2
if middle_points > 0:
    # Create monotonically decreasing values that sum to 1
    # Use a geometric sequence with ratio r
    r = 0.6  # Adjust this to control the rate of decrease
    y_values = np.array([r**i for i in range(middle_points)])
    y_values = y_values / np.sum(y_values)  # Normalize to sum to 1
    
    # Distribute x-values linearly between 0 and distance
    x_values = np.linspace(0, distance, middle_points + 2)[1:-1]
    
    for i in range(middle_points):
        control_points[i + 1] = [x_values[i], y_values[i]]
'''

control_points[1] = [1, 0.6]
control_points[2] = [2, 0.30]
control_points[3] = [4, 0.10]

# Create the B-spline
spline = BSpline(knots, control_points, degree)

# Generate points for plotting
t = np.linspace(knots[degree], knots[-degree-1], 1000)
spline_points = spline(t)

# Plotting
plt.style.use('dark_background')
plt.figure(figsize=(10, 6))
plt.plot(spline_points[:, 0], spline_points[:, 1], 'w-', linewidth=2, label='B-spline')
plt.plot(control_points[:, 0], control_points[:, 1], 'o', color='orange', markersize=10, label='Control points')
plt.xlabel('Distance', fontsize=14)
plt.ylabel('Risk', fontsize=14)
#plt.title(f'Degree {degree} B-spline with Linearly Spaced Knots')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.ylim(0, 1)
plt.show()

print(f"Control points:")
for i, point in enumerate(control_points):
    print(f"P{i}: ({point[0]:.3f}, {point[1]:.3f})")

print(f"\nSum of middle control point y-values: {np.sum(control_points[1:-1, 1]):.6f}")

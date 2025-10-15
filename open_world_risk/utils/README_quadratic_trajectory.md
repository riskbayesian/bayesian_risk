# 3D Quadratic Trajectory Calculator

This module provides functionality to compute a quadratic trajectory in 3D space that passes through exactly 3 given points, and evaluates N equally spaced points along that trajectory.

## Features

- **Quadratic Interpolation**: Computes quadratic curves in each dimension (x, y, z) that pass through 3 given points
- **Equally Spaced Sampling**: Generates N equally spaced points along the trajectory
- **3D Visualization**: Optional plotting functionality with matplotlib
- **Trajectory Analysis**: Calculate trajectory length and evaluate at specific parameter values
- **Flexible API**: Both object-oriented and convenience function interfaces

## Mathematical Background

The trajectory is parameterized as a quadratic curve in each dimension:
- **X(t) = a_x * t² + b_x * t + c_x**
- **Y(t) = a_y * t² + b_y * t + c_y**
- **Z(t) = a_z * t² + b_z * t + c_z**

Where t ranges from 0 to 2, with:
- t = 0: First point
- t = 1: Second point  
- t = 2: Third point

The coefficients are computed by solving the linear system formed by the 3 points.

## Usage

### Basic Usage

```python
from quadratic_trajectory import QuadraticTrajectory3D, compute_quadratic_trajectory

# Define 3 points in 3D space
points = [
    (0, 0, 0),    # Start point
    (2, 3, 1),    # Middle point
    (5, 1, 4)     # End point
]

# Method 1: Using the class
trajectory = QuadraticTrajectory3D(points)
trajectory_points = trajectory.get_equally_spaced_points(N=50)

# Method 2: Using the convenience function
trajectory_points = compute_quadratic_trajectory(points, N=50)
```

### Advanced Usage

```python
# Create trajectory object
trajectory = QuadraticTrajectory3D(points)

# Get trajectory information
length = trajectory.get_trajectory_length()
coefficients = trajectory.coefficients

# Evaluate at specific parameter values
point_at_05 = trajectory.evaluate_at_t(0.5)  # Halfway between first and second point
point_at_15 = trajectory.evaluate_at_t(1.5)  # Halfway between second and third point

# Plot the trajectory (requires matplotlib)
trajectory.plot_trajectory(N=100)
```

## API Reference

### QuadraticTrajectory3D Class

#### Constructor
```python
QuadraticTrajectory3D(points: List[Tuple[float, float, float]])
```
- **points**: List of exactly 3 tuples, each containing (x, y, z) coordinates

#### Methods

- `get_equally_spaced_points(N: int) -> np.ndarray`
  - Returns N equally spaced points along the trajectory
  - Returns array of shape (N, 3)

- `evaluate_at_t(t: float) -> Tuple[float, float, float]`
  - Evaluates the trajectory at parameter value t
  - Returns (x, y, z) coordinates

- `get_trajectory_length(N: int = 1000) -> float`
  - Computes approximate length of the trajectory
  - Uses N points for approximation

- `plot_trajectory(N: int = 100, show_original_points: bool = True)`
  - Plots the trajectory in 3D (requires matplotlib)
  - Shows original 3 points as red dots

#### Properties

- `coefficients`: Array of shape (3, 3) containing quadratic coefficients [a, b, c] for each dimension
- `points`: Original 3 points as numpy array

### Convenience Function

```python
compute_quadratic_trajectory(points: List[Tuple[float, float, float]], N: int = 100) -> np.ndarray
```
- **points**: List of 3 tuples with (x, y, z) coordinates
- **N**: Number of equally spaced points to return
- **Returns**: Array of shape (N, 3) with trajectory points

## Examples

### Example 1: Basic Trajectory
```python
points = [(0, 0, 0), (2, 3, 1), (5, 1, 4)]
trajectory = QuadraticTrajectory3D(points)
points_50 = trajectory.get_equally_spaced_points(50)
```

### Example 2: Different Trajectory Types
```python
# Linear-like trajectory
linear_points = [(0, 0, 0), (1, 1, 1), (2, 2, 2)]

# Curved trajectory  
curved_points = [(0, 0, 0), (2, 4, 1), (0, 0, 2)]

# Complex 3D curve
complex_points = [(0, 0, 0), (3, 2, 5), (1, 4, 1)]
```

### Example 3: Analysis
```python
trajectory = QuadraticTrajectory3D(points)

# Get coefficients
print(f"X coefficients: {trajectory.coefficients[0]}")
print(f"Y coefficients: {trajectory.coefficients[1]}")
print(f"Z coefficients: {trajectory.coefficients[2]}")

# Calculate length
length = trajectory.get_trajectory_length()
print(f"Trajectory length: {length:.3f}")
```

## Dependencies

- `numpy`: For numerical computations
- `matplotlib` (optional): For 3D plotting
- `typing`: For type hints

## Files

- `quadratic_trajectory.py`: Main implementation
- `test_quadratic_trajectory.py`: Comprehensive tests
- `example_usage.py`: Usage examples
- `README_quadratic_trajectory.md`: This documentation

## Running the Examples

```bash
# Run tests
python src/test_quadratic_trajectory.py

# Run examples
python src/example_usage.py

# Run main example with plotting
python src/quadratic_trajectory.py
```

## Notes

- The trajectory is guaranteed to pass exactly through the 3 given points
- The parameter t ranges from 0 to 2, corresponding to the 3 points
- The trajectory is smooth and continuous
- For plotting, matplotlib with 3D support is required
- The implementation uses numpy's linear algebra solver for efficiency 
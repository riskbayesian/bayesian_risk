#!/usr/bin/env python3
"""
Test script for the quadratic trajectory functionality.
"""

import numpy as np
from open_world_risk.utils.quadratic_trajectory import QuadraticTrajectory3D, compute_quadratic_trajectory


def test_basic_trajectory():
    """Test basic trajectory computation."""
    print("=== Test 1: Basic Trajectory ===")
    
    # Define 3 points in 3D
    points = [
        (0, 0, 0),    # Start point
        (2, 3, 1),    # Middle point  
        (5, 1, 4)     # End point
    ]
    
    # Create trajectory
    trajectory = QuadraticTrajectory3D(points)
    
    # Get 20 equally spaced points
    N = 20
    trajectory_points = trajectory.get_equally_spaced_points(N)
    
    print(f"Original points: {points}")
    print(f"Number of equally spaced points: {N}")
    print(f"First 5 trajectory points:")
    for i in range(5):
        print(f"  Point {i+1}: {trajectory_points[i]}")
    
    # Verify that the trajectory passes through the original points
    print(f"\nVerifying trajectory passes through original points:")
    for i, point in enumerate(points):
        t = i  # t = 0, 1, 2 for the 3 points
        computed_point = trajectory.evaluate_at_t(t)
        print(f"  Original P{i+1}: {point}")
        print(f"  Computed at t={t}: {computed_point}")
        print(f"  Difference: {np.array(point) - np.array(computed_point)}")
    
    return trajectory


def test_convenience_function():
    """Test the convenience function."""
    print("\n=== Test 2: Convenience Function ===")
    
    points = [
        (1, 1, 1),
        (3, 2, 4),
        (6, 0, 2)
    ]
    
    N = 15
    trajectory_points = compute_quadratic_trajectory(points, N)
    
    print(f"Using convenience function with {N} points:")
    print(f"Trajectory shape: {trajectory_points.shape}")
    print(f"First 3 points: {trajectory_points[:3]}")


def test_different_scenarios():
    """Test different trajectory scenarios."""
    print("\n=== Test 3: Different Scenarios ===")
    
    # Scenario 1: Linear-like trajectory
    print("Scenario 1: Near-linear trajectory")
    points1 = [
        (0, 0, 0),
        (1, 1, 1),
        (2, 2, 2)
    ]
    traj1 = QuadraticTrajectory3D(points1)
    print(f"Trajectory length: {traj1.get_trajectory_length():.3f}")
    
    # Scenario 2: Curved trajectory
    print("\nScenario 2: Curved trajectory")
    points2 = [
        (0, 0, 0),
        (2, 4, 1),
        (0, 0, 2)
    ]
    traj2 = QuadraticTrajectory3D(points2)
    print(f"Trajectory length: {traj2.get_trajectory_length():.3f}")
    
    # Scenario 3: Complex 3D curve
    print("\nScenario 3: Complex 3D curve")
    points3 = [
        (0, 0, 0),
        (3, 2, 5),
        (1, 4, 1)
    ]
    traj3 = QuadraticTrajectory3D(points3)
    print(f"Trajectory length: {traj3.get_trajectory_length():.3f}")


def test_coefficients():
    """Test coefficient computation."""
    print("\n=== Test 4: Coefficient Analysis ===")
    
    points = [
        (0, 0, 0),
        (2, 3, 1),
        (5, 1, 4)
    ]
    
    trajectory = QuadraticTrajectory3D(points)
    
    print("Quadratic coefficients for each dimension:")
    print(f"X: a={trajectory.coefficients[0, 0]:.3f}, b={trajectory.coefficients[0, 1]:.3f}, c={trajectory.coefficients[0, 2]:.3f}")
    print(f"Y: a={trajectory.coefficients[1, 0]:.3f}, b={trajectory.coefficients[1, 1]:.3f}, c={trajectory.coefficients[1, 2]:.3f}")
    print(f"Z: a={trajectory.coefficients[2, 0]:.3f}, b={trajectory.coefficients[2, 1]:.3f}, c={trajectory.coefficients[2, 2]:.3f}")
    
    # Verify the quadratic equations
    print("\nVerifying quadratic equations:")
    for i, point in enumerate(points):
        t = i
        x, y, z = point
        a_x, b_x, c_x = trajectory.coefficients[0]
        a_y, b_y, c_y = trajectory.coefficients[1]
        a_z, b_z, c_z = trajectory.coefficients[2]
        
        x_calc = a_x * t**2 + b_x * t + c_x
        y_calc = a_y * t**2 + b_y * t + c_y
        z_calc = a_z * t**2 + b_z * t + c_z
        
        print(f"Point {i+1} (t={t}):")
        print(f"  Original: ({x}, {y}, {z})")
        print(f"  Calculated: ({x_calc:.3f}, {y_calc:.3f}, {z_calc:.3f})")


if __name__ == "__main__":
    # Run all tests
    trajectory = test_basic_trajectory()
    test_convenience_function()
    test_different_scenarios()
    test_coefficients()
    
    print("\n=== All tests completed ===")
    print("You can now use the trajectory object to plot or analyze the trajectory further.") 
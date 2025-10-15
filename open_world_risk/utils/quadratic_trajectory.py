import numpy as np
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class QuadraticTrajectory3D:
    """
    Computes a quadratic trajectory in 3D that passes through 3 given points
    and evaluates N equally spaced points along that trajectory with velocities.
    """
    
    def __init__(self, points: List[Tuple[float, float, float]]):
        """
        Initialize with 3 points in 3D space.
        
        Args:
            points: List of 3 tuples, each containing (x, y, z) coordinates
        """
        if len(points) != 3:
            raise ValueError("Exactly 3 points are required")
        
        self.points = np.array(points)
        self.coefficients = self._compute_quadratic_coefficients()
    
    def _compute_quadratic_coefficients(self) -> np.ndarray:
        """
        Compute quadratic coefficients for each dimension (x, y, z).
        
        For each dimension, we solve the system:
        a*t² + b*t + c = coordinate
        for t = 0, 1, 2 (representing the 3 points)
        
        Returns:
            Array of shape (3, 3) where each row contains [a, b, c] for x, y, z dimensions
        """
        # We'll parameterize the trajectory with t = 0, 1, 2 for the 3 points
        t_values = np.array([0, 1, 2])
        
        # Create the coefficient matrix for the quadratic system
        # For each point: a*t² + b*t + c = coordinate
        A = np.array([[t**2, t, 1] for t in t_values])
        
        coefficients = np.zeros((3, 3))  # 3 dimensions, 3 coefficients each
        
        # Solve for each dimension (x, y, z)
        for dim in range(3):
            b = self.points[:, dim]
            coeffs = np.linalg.solve(A, b)
            coefficients[dim] = coeffs
        
        return coefficients
    
    def evaluate_at_t(self, t: float) -> Tuple[float, float, float]:
        """
        Evaluate the trajectory at parameter t.
        
        Args:
            t: Parameter value (typically 0 to 2 for the original 3 points)
            
        Returns:
            Tuple of (x, y, z) coordinates
        """
        # For each dimension: a*t² + b*t + c
        result = np.zeros(3)
        for dim in range(3):
            a, b, c = self.coefficients[dim]
            result[dim] = a * t**2 + b * t + c
        
        return tuple(result)
    
    def evaluate_velocity_at_t(self, t: float) -> Tuple[float, float, float]:
        """
        Evaluate the velocity (derivative) at parameter t.
        
        Args:
            t: Parameter value (typically 0 to 2 for the original 3 points)
            
        Returns:
            Tuple of (vx, vy, vz) velocities
        """
        # For each dimension: velocity = 2*a*t + b (derivative of a*t² + b*t + c)
        result = np.zeros(3)
        for dim in range(3):
            a, b, c = self.coefficients[dim]
            result[dim] = 2 * a * t + b
        
        return tuple(result)
    
    def get_equally_spaced_points(self, N: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get N equally spaced points along the trajectory with their velocities.
        
        Args:
            N: Number of points to evaluate
            
        Returns:
            Tuple of (positions, velocities) where:
            - positions: Array of shape (N, 3) containing the (x, y, z) coordinates
            - velocities: Array of shape (N, 3) containing the (vx, vy, vz) velocities
        """
        # Parameterize from t=0 to t=2 (the range of our 3 original points)
        t_values = np.linspace(0, 2, N)
        
        positions = np.zeros((N, 3))
        velocities = np.zeros((N, 3))
        
        for i, t in enumerate(t_values):
            # Get position
            x, y, z = self.evaluate_at_t(t)
            positions[i] = [x, y, z]
            
            # Get velocity
            vx, vy, vz = self.evaluate_velocity_at_t(t)
            velocities[i] = [vx, vy, vz]
        
        return positions, velocities
    
    def get_trajectory_length(self, N: int = 1000) -> float:
        """
        Compute the approximate length of the trajectory.
        
        Args:
            N: Number of points to use for approximation
            
        Returns:
            Approximate length of the trajectory
        """
        positions, _ = self.get_equally_spaced_points(N)
        distances = np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1))
        return np.sum(distances)
    
    def plot_trajectory(self, N: int = 100, show_original_points: bool = True, show_velocities: bool = False):
        """
        Plot the trajectory in 3D with optional velocity vectors.
        
        Args:
            N: Number of points to plot along the trajectory
            show_original_points: Whether to highlight the original 3 points
            show_velocities: Whether to show velocity vectors at sample points
        """
        positions, velocities = self.get_equally_spaced_points(N)
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot the trajectory
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                'b-', linewidth=2, label='Quadratic Trajectory')
        
        if show_velocities:
            # Show velocity vectors at every 10th point
            step = max(1, N // 20)  # Show at most 20 velocity vectors
            for i in range(0, N, step):
                pos = positions[i]
                vel = velocities[i]
                # Scale velocity for visualization
                vel_scaled = vel * 0.1  # Scale factor for visualization
                ax.quiver(pos[0], pos[1], pos[2], 
                         vel_scaled[0], vel_scaled[1], vel_scaled[2],
                         color='red', alpha=0.7, length=0.5)
        
        if show_original_points:
            # Plot the original 3 points
            ax.scatter(self.points[:, 0], self.points[:, 1], self.points[:, 2], 
                      c='red', s=100, marker='o', label='Original Points')
            
            # Add point labels
            for i, point in enumerate(self.points):
                ax.text(point[0], point[1], point[2], f'P{i+1}', fontsize=12)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Quadratic Trajectory with Velocities' if show_velocities else '3D Quadratic Trajectory')
        ax.legend()
        
        plt.tight_layout()
        plt.show()


def compute_quadratic_trajectory(points: List[Tuple[float, float, float]], 
                                N: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to compute quadratic trajectory and return N equally spaced points with velocities.
    
    Args:
        points: List of 3 tuples, each containing (x, y, z) coordinates
        N: Number of equally spaced points to return
        
    Returns:
        Tuple of (positions, velocities) where:
        - positions: Array of shape (N, 3) containing the (x, y, z) coordinates
        - velocities: Array of shape (N, 3) containing the (vx, vy, vz) velocities
    """
    trajectory = QuadraticTrajectory3D(points)
    return trajectory.get_equally_spaced_points(N)


# Example usage and testing
if __name__ == "__main__":
    # Example: Create a trajectory through 3 points
    points = [
        (0, 0, 0),    # Start point
        (2, 3, 1),    # Middle point
        (5, 1, 4)     # End point
    ]
    
    # Create trajectory object
    trajectory = QuadraticTrajectory3D(points)
    
    # Get 50 equally spaced points with velocities
    N = 50
    positions, velocities = trajectory.get_equally_spaced_points(N)
    
    print(f"Original 3 points:")
    for i, point in enumerate(points):
        print(f"P{i+1}: {point}")
    
    print(f"\nQuadratic coefficients:")
    print(f"X: a={trajectory.coefficients[0, 0]:.3f}, b={trajectory.coefficients[0, 1]:.3f}, c={trajectory.coefficients[0, 2]:.3f}")
    print(f"Y: a={trajectory.coefficients[1, 0]:.3f}, b={trajectory.coefficients[1, 1]:.3f}, c={trajectory.coefficients[1, 2]:.3f}")
    print(f"Z: a={trajectory.coefficients[2, 0]:.3f}, b={trajectory.coefficients[2, 1]:.3f}, c={trajectory.coefficients[2, 2]:.3f}")
    
    print(f"\nFirst 10 of {N} equally spaced points with velocities:")
    for i in range(min(10, N)):
        print(f"Point {i+1}: pos={positions[i]}, vel={velocities[i]}")
    
    print(f"\nTrajectory length: {trajectory.get_trajectory_length():.3f}")
    
    # Plot the trajectory with velocities
    trajectory.plot_trajectory(N=100, show_velocities=True) 
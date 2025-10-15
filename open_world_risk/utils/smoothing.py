import torch
import numpy as np
import cvxpy as cp
from typing import Union, Tuple, Optional
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def sample_equidistant_points(trajectory, n_samples=50):
    """
    Sample roughly equidistant points from a trajectory.
    
    Args:
        trajectory: Shape (n_points, dim) - the trajectory to sample from
        n_samples: Number of points to sample
    
    Returns:
        sampled_points: Shape (n_samples, dim) - equidistant samples
        sampled_indices: Indices of the sampled points in the original trajectory
    """
    n_points = trajectory.shape[0]
    
    if n_samples >= n_points:
        return trajectory, torch.arange(n_points)
    
    # Calculate cumulative distances along the trajectory
    distances = torch.zeros(n_points, device=trajectory.device)
    for i in range(1, n_points):
        distances[i] = distances[i-1] + torch.norm(trajectory[i] - trajectory[i-1])
    
    total_distance = distances[-1]
    
    # Sample equidistant points along the cumulative distance
    target_distances = torch.linspace(0, total_distance, n_samples, device=trajectory.device)
    
    sampled_indices = torch.zeros(n_samples, dtype=torch.long, device=trajectory.device)
    sampled_points = torch.zeros(n_samples, trajectory.shape[1], device=trajectory.device)
    
    for i, target_dist in enumerate(target_distances):
        # Find the closest point in the cumulative distance
        idx = torch.argmin(torch.abs(distances - target_dist))
        sampled_indices[i] = idx
        sampled_points[i] = trajectory[idx]
    
    return sampled_points, sampled_indices

def bernstein_polynomial(n: int, k: int, t: np.ndarray) -> np.ndarray:
    """
    Compute the Bernstein polynomial B_{n,k}(t).
    
    Args:
        n: Degree of the polynomial
        k: Index of the Bernstein polynomial
        t: Parameter values (0 to 1)
    
    Returns:
        Bernstein polynomial values
    """
    if k < 0 or k > n:
        return np.zeros_like(t)
    
    # Compute binomial coefficient
    binom = np.prod(np.arange(k + 1, n + 1)) / np.prod(np.arange(1, n - k + 1))
    
    return binom * (t ** k) * ((1 - t) ** (n - k))


def bezier_curve(control_points: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Compute points on a Bezier curve.
    
    Args:
        control_points: Shape (n_control_points, dim) - control points of the Bezier curve
        t: Parameter values (0 to 1) - shape (n_points,)
    
    Returns:
        Points on the Bezier curve - shape (n_points, dim)
    """
    n_control_points = control_points.shape[0]
    n = n_control_points - 1  # Degree of the curve
    dim = control_points.shape[1]
    
    # Compute Bernstein polynomials for all k values
    bernstein_values = np.column_stack([bernstein_polynomial(n, k, t) for k in range(n_control_points)])
    
    # Compute the curve points
    curve_points = np.zeros((t.shape[0], dim))
    for i in range(n_control_points):
        curve_points += bernstein_values[:, i:i+1] * control_points[i:i+1]
    
    return curve_points


def optimize_bezier_curve_cvxpy(
    data_points: Union[torch.Tensor, np.ndarray],
    weights: Union[torch.Tensor, np.ndarray],
    start_point: Union[torch.Tensor, np.ndarray],
    end_point: Union[torch.Tensor, np.ndarray],
    grads: Optional[Union[torch.Tensor, np.ndarray]] = None,
    n_control_points: int = 4,
    regularization: float = 0.0,
    initial_control_points: Optional[Union[torch.Tensor, np.ndarray]] = None,
    solver: str = 'OSQP'
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Optimize a smooth Bezier curve to fit weighted data points using CVXPY.
    
    Args:
        data_points: Shape (n_points, dim) - target data points (2D or 3D)
        weights: Shape (n_points,) - weights for each data point
        n_control_points: Number of control points for the Bezier curve
        regularization: Regularization parameter for smoothness
        initial_control_points: Initial guess for control points (optional)
        solver: CVXPY solver to use ('OSQP', 'SCS', 'ECOS', etc.)
    
    Returns:
        Tuple of (optimized_control_points, curve_points, final_loss)
    """
    # Convert to numpy if needed
    if isinstance(data_points, torch.Tensor):
        data_points = data_points.cpu().numpy()
    if isinstance(weights, torch.Tensor):
        weights = weights.cpu().numpy()
    if initial_control_points is not None and isinstance(initial_control_points, torch.Tensor):
        initial_control_points = initial_control_points.cpu().numpy()
    
    n_points, dim = data_points.shape
    
    # Generate parameter values t for the data points
    t_values = np.linspace(0, 1, n_points)
    
    # Compute Bernstein polynomial matrix
    n = n_control_points - 1  # Degree of the curve
    B = np.column_stack([bernstein_polynomial(n, k, t_values) for k in range(n_control_points)])
    
    # Create CVXPY variables for control points
    control_points = cp.Variable((n_control_points, dim))
    
    # Data fitting objective (weighted least squares)
    # B @ control_points gives us the curve points
    curve_points = B @ control_points
    fitting_loss = cp.sum_squares(cp.multiply(weights[:, np.newaxis], curve_points - data_points))

    # Smoothness regularization (penalize large differences between consecutive control points)
    smoothness_loss = 0.0
    if regularization > 0:
        for i in range(1, n_control_points):
            smoothness_loss += cp.sum_squares(control_points[i] - control_points[i-1])
        smoothness_loss *= regularization
    
    # Total objective
    objective = cp.Minimize(fitting_loss + smoothness_loss)
    constraint = [
        control_points[0] == start_point,
        control_points[-1] == end_point
    ]

    if grads is not None:
        diff = curve_points - data_points
        diff_grads = cp.sum(cp.multiply(diff, grads), axis=1)
        constraint += [diff_grads >= 0.]
    
    # Create and solve the problem
    problem = cp.Problem(objective, constraints=constraint)
    
    # Set initial value if provided
    if initial_control_points is not None:
        control_points.value = initial_control_points
    
    # Solve the problem
    try:
        result = problem.solve(solver=solver, verbose=False)
        if problem.status not in ['optimal', 'optimal_inaccurate']:
            print(f"Warning: Solver status: {problem.status}")
    except Exception as e:
        print(f"Error solving with {solver}: {e}")
        # Try with a different solver
        try:
            result = problem.solve(solver='SCS', verbose=False)
            print(f"Solved with SCS, status: {problem.status}")
        except Exception as e2:
            print(f"Error with SCS solver: {e2}")
            raise
    
    # Extract results
    optimized_control_points = control_points.value
    
    # Compute final curve points
    final_curve_points = bezier_curve(optimized_control_points, t_values)
    
    # Compute final loss
    final_loss = problem.value
    
    return optimized_control_points, final_curve_points, final_loss


def optimize_bezier_curve(
    data_points: torch.Tensor,
    weights: torch.Tensor,
    n_control_points: int = 4,
    regularization: float = 0.01,
    max_iterations: int = 1000,
    learning_rate: float = 0.01,
    tolerance: float = 1e-6,
    initial_control_points: Optional[torch.Tensor] = None,
    device: str = 'cuda'
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Optimize a smooth Bezier curve to fit weighted data points using CVXPY.
    
    Args:
        data_points: Shape (n_points, dim) - target data points (2D or 3D)
        weights: Shape (n_points,) - weights for each data point
        n_control_points: Number of control points for the Bezier curve
        regularization: Regularization parameter for smoothness
        max_iterations: Maximum number of optimization iterations (ignored for CVXPY)
        learning_rate: Learning rate for gradient descent (ignored for CVXPY)
        tolerance: Convergence tolerance (ignored for CVXPY)
        initial_control_points: Initial guess for control points (optional)
        device: Device to run optimization on (ignored for CVXPY)
    
    Returns:
        Tuple of (optimized_control_points, curve_points, final_loss)
    """
    # Use CVXPY optimization
    optimized_control_points, final_curve_points, final_loss = optimize_bezier_curve_cvxpy(
        data_points=data_points,
        weights=weights,
        n_control_points=n_control_points,
        regularization=regularization,
        initial_control_points=initial_control_points
    )
    
    # Convert back to torch tensors
    optimized_control_points_torch = torch.from_numpy(optimized_control_points).to(device)
    final_curve_points_torch = torch.from_numpy(final_curve_points).to(device)
    
    return optimized_control_points_torch, final_curve_points_torch, final_loss


def visualize_bezier_fit(
    data_points: torch.Tensor,
    control_points: torch.Tensor,
    curve_points: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    title: str = "Bezier Curve Fit"
) -> None:
    """
    Visualize the Bezier curve fit to the data points.
    
    Args:
        data_points: Original data points
        control_points: Optimized control points
        curve_points: Points on the fitted curve
        weights: Weights used for fitting (optional, for visualization)
        title: Plot title
    """
    dim = data_points.shape[1]
    
    if dim == 2:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Plot data points with size based on weights
        if weights is not None:
            sizes = 100 * weights / weights.max()
            ax.scatter(data_points[:, 0].cpu(), data_points[:, 1].cpu(), 
                      c='red', s=sizes.cpu(), alpha=0.7, label='Data Points')
        else:
            ax.scatter(data_points[:, 0].cpu(), data_points[:, 1].cpu(), 
                      c='red', s=50, alpha=0.7, label='Data Points')
        
        # Plot control points
        ax.scatter(control_points[:, 0].cpu(), control_points[:, 1].cpu(), 
                  c='blue', s=100, marker='s', label='Control Points')
        
        # Plot control polygon
        ax.plot(control_points[:, 0].cpu(), control_points[:, 1].cpu(), 
               'b--', alpha=0.5, label='Control Polygon')
        
        # Plot fitted curve
        ax.plot(curve_points[:, 0].cpu(), curve_points[:, 1].cpu(), 
               'g-', linewidth=2, label='Fitted Curve')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title(title)
        ax.set_aspect('equal')
        
    elif dim == 3:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot data points with size based on weights
        if weights is not None:
            sizes = 100 * weights / weights.max()
            ax.scatter(data_points[:, 0].cpu(), data_points[:, 1].cpu(), data_points[:, 2].cpu(),
                      c='red', s=sizes.cpu(), alpha=0.7, label='Data Points')
        else:
            ax.scatter(data_points[:, 0].cpu(), data_points[:, 1].cpu(), data_points[:, 2].cpu(),
                      c='red', s=50, alpha=0.7, label='Data Points')
        
        # Plot control points
        ax.scatter(control_points[:, 0].cpu(), control_points[:, 1].cpu(), control_points[:, 2].cpu(),
                  c='blue', s=100, marker='s', label='Control Points')
        
        # Plot control polygon
        ax.plot(control_points[:, 0].cpu(), control_points[:, 1].cpu(), control_points[:, 2].cpu(),
               'b--', alpha=0.5, label='Control Polygon')
        
        # Plot fitted curve
        ax.plot(curve_points[:, 0].cpu(), curve_points[:, 1].cpu(), curve_points[:, 2].cpu(),
               'g-', linewidth=2, label='Fitted Curve')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        ax.set_title(title)
    
    plt.tight_layout()
    plt.show()


# Example usage function
def example_bezier_optimization():
    """
    Example demonstrating the Bezier curve optimization with CVXPY.
    """
    # Generate sample data points (2D example)
    torch.manual_seed(42)
    
    # Create a noisy spiral pattern
    t = torch.linspace(0, 4*torch.pi, 100)
    true_x = t * torch.cos(t)
    true_y = t * torch.sin(t)
    
    # Add noise
    noise_level = 0.1
    noisy_x = true_x + noise_level * torch.randn_like(true_x)
    noisy_y = true_y + noise_level * torch.randn_like(true_y)
    
    data_points = torch.stack([noisy_x, noisy_y], dim=1)
    
    # Create weights (higher weight for points closer to the center)
    distances = torch.norm(data_points, dim=1)
    weights = torch.exp(-distances / 2.0)  # Exponential decay from center
    
    print("Optimizing Bezier curve with CVXPY...")
    
    # Optimize Bezier curve using CVXPY
    control_points, curve_points, final_loss = optimize_bezier_curve(
        data_points=data_points,
        weights=weights,
        n_control_points=8,
        regularization=0.01
    )
    
    print(f"Final loss: {final_loss:.6f}")
    
    # Visualize results
    visualize_bezier_fit(data_points, control_points, curve_points, weights, 
                        "Bezier Curve Fit with CVXPY")


if __name__ == "__main__":
    example_bezier_optimization()

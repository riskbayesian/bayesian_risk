import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from sklearn.linear_model import Ridge
import heapq
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import concurrent.futures
import threading

def distance_cost_field(item_pos, weight, grid_size=50, sigma=5.0):
    """Define distance-based cost fields (Gaussian decay)"""
    x, y = np.meshgrid(np.arange(grid_size), np.arange(grid_size), indexing='ij')
    dist_sq = (x - item_pos[0])**2 + (y - item_pos[1])**2
    return weight * np.exp(-dist_sq / (2 * sigma**2))

def heuristic(a, b):
    """Heuristic function for A* algorithm"""
    return np.linalg.norm(np.array(a) - np.array(b))

def a_star(cost_map, start, goal, grid_size=50):
    """A* pathfinding algorithm - optimized version"""
    neighbors = [(0,1),(1,0),(0,-1),(-1,0)]
    close_set = set()
    came_from = {}
    gscore = {start:0}
    fscore = {start:heuristic(start, goal)}
    oheap = []

    heapq.heappush(oheap, (fscore[start], start))

    # Early termination if start and goal are the same
    if start == goal:
        return [start]

    while oheap:
        _, current = heapq.heappop(oheap)
        if current == goal:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            data.append(start)
            return data[::-1]

        close_set.add(current)
        for i, j in neighbors:
            neighbor = (current[0]+i, current[1]+j)
            if 0 <= neighbor[0] < grid_size and 0 <= neighbor[1] < grid_size:
                tentative_g_score = gscore[current] + cost_map[neighbor]
                if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, float('inf')):
                    continue
                if tentative_g_score < gscore.get(neighbor, float('inf')) or neighbor not in [i[1] for i in oheap]:
                    came_from[neighbor] = current
                    gscore[neighbor] = tentative_g_score
                    fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(oheap, (fscore[neighbor], neighbor))
    return []


class CostFunctionLearner(nn.Module):
    """Neural network to learn weight and sigma parameters for each item"""
    
    def __init__(self, num_items, grid_size=50, hidden_size=64):
        super(CostFunctionLearner, self).__init__()
        self.num_items = num_items
        self.grid_size = grid_size
        
        # Feature dimension: distances from each path point to each item
        # We'll process this per path, so input_dim will be dynamic
        self.hidden_size = hidden_size
        
        # Traditional neural network architecture
        self.feature_encoder = nn.Sequential(
            nn.Linear(num_items, hidden_size),  # num_items distances as input
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Output layers to predict weight and sigma for each item
        self.weight_predictor = nn.Linear(hidden_size // 2, num_items)
        self.sigma_predictor = nn.Linear(hidden_size // 2, num_items)
        
        # Initialize weights properly
        self._initialize_weights()
        
        # Activation functions to ensure positive outputs
        self.weight_activation = nn.Softplus()  # Ensures positive weights
        self.sigma_activation = nn.Softplus()   # Ensures positive sigmas
        
    def _initialize_weights(self):
        """Initialize weights to prevent NaN issues"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
    def forward(self, paths, items):
        """Compute predicted total cost for each path - optimized version"""
        batch_size = len(paths)
        device = next(self.parameters()).device  # Get device from model parameters
        total_costs = torch.zeros(batch_size, dtype=torch.float32, device=device)
        
        # Pre-compute all distances for efficiency
        all_distances = []
        for path in paths:
            path_array = np.array(path)
            items_array = np.array(items)
            # Vectorized distance computation
            distances = np.linalg.norm(path_array[:, np.newaxis, :] - items_array[np.newaxis, :, :], axis=2)
            all_distances.append(distances)
        
        # Process each path separately (but with pre-computed distances)
        all_weights = []
        all_sigmas = []
        
        for i, distances in enumerate(all_distances):
            # Convert to tensor: shape (num_points, num_items)
            distances_tensor = torch.tensor(distances, dtype=torch.float32, device=device)
            
            # Encode features through hidden layers
            encoded = self.feature_encoder(distances_tensor)  # Shape: (num_points, hidden_size)
            
            # Predict weights and sigmas for each path point
            weights = self.weight_activation(self.weight_predictor(encoded))  # Shape: (num_points, num_items)
            sigmas = self.sigma_activation(self.sigma_predictor(encoded))    # Shape: (num_points, num_items)
            
            # Clamp parameters to reasonable ranges
            weights = torch.clamp(weights, min=0.1, max=20.0)
            sigmas = torch.clamp(sigmas, min=0.5, max=20.0)
            
            # Compute cost contribution at each point
            # Shape: (num_points, num_items)
            cost_contributions = weights * torch.exp(-distances_tensor**2 / (2 * sigmas**2))
            
            # Take maximum cost at each point (like the ground truth)
            max_costs_per_point = torch.max(cost_contributions, dim=1)[0]  # Max across items
            
            # Sum the maximum costs along the path
            path_cost = torch.sum(max_costs_per_point)
            total_costs[i] = path_cost
            
            # Store average weights and sigmas for this path
            all_weights.append(weights.mean(dim=0))  # Average across points
            all_sigmas.append(sigmas.mean(dim=0))    # Average across points
        
        # Stack all path averages
        weights = torch.stack(all_weights)  # Shape: (batch_size, num_items)
        sigmas = torch.stack(all_sigmas)    # Shape: (batch_size, num_items)
        
        return total_costs, weights, sigmas


def train_cost_function_model(paths, items, num_epochs=1000, lr=0.0001):
    """Train the neural network to learn cost function parameters from optimal paths only"""
    # Set up device (MPS for Apple Silicon, CPU fallback)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = CostFunctionLearner(len(items), hidden_size=64)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Training loop with progress bar
    with tqdm(total=num_epochs, desc="Training") as pbar:
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            
            # Forward pass - predict costs for each path
            predicted_costs, weights, sigmas = model(paths, items)
            
            # Check for NaN in predictions
            if torch.isnan(predicted_costs).any():
                print(f"\nNaN detected in predictions at epoch {epoch}")
                break
            
            # Loss: Encourage paths to have similar costs (they were all chosen as optimal)
            # The idea is that optimal paths should have similar total costs
            mean_cost = torch.mean(predicted_costs)
            cost_variance = torch.var(predicted_costs)
            
            # Loss: minimize variance in path costs (all paths should be similarly optimal)
            loss = cost_variance
            
            # Add regularization to encourage reasonable parameter scales
            weight_penalty = 0.001 * torch.sum(torch.abs(weights - 5.0))  # Encourage weights near 5.0
            sigma_penalty = 0.001 * torch.sum(torch.abs(sigmas - 5.0))    # Encourage sigmas near 5.0
            
            total_loss = loss + weight_penalty + sigma_penalty
            
            # Check for NaN in loss
            if torch.isnan(total_loss):
                print(f"\nNaN loss detected at epoch {epoch}")
                break
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Update progress bar with current loss
            pbar.set_postfix({
                'Loss': f'{total_loss.item():.4f}',
                'Var': f'{cost_variance.item():.4f}',
                'Mean': f'{mean_cost.item():.4f}',
                'W1': f'{weights.mean(dim=0)[0].item():.2f}',
                'W2': f'{weights.mean(dim=0)[1].item():.2f}',
                'W3': f'{weights.mean(dim=0)[2].item():.2f}'
            })
            pbar.update(1)
    
    # Final training summary
    print(f"\nTraining completed after {num_epochs} epochs")
    print(f"Final loss: {total_loss.item():.4f}")
    
    return model



def generate_single_path(cost_field, grid_size):
    """Generate a single path - used for parallel processing"""
    start = (np.random.randint(0, grid_size), np.random.randint(0, grid_size))
    goal = (np.random.randint(0, grid_size), np.random.randint(0, grid_size))
    
    # Skip if start and goal are too close (likely to fail)
    if np.linalg.norm(np.array(start) - np.array(goal)) < 5:
        return None
    
    path = a_star(cost_field, start, goal, grid_size)
    return path if path else None

def generate_simple_paths(cost_field, grid_size=50, num_paths=10):
    """Generate simple straight-line paths for faster training"""
    paths = []
    for _ in range(num_paths):
        start = (np.random.randint(0, grid_size), np.random.randint(0, grid_size))
        goal = (np.random.randint(0, grid_size), np.random.randint(0, grid_size))
        
        # Create a simple path (not optimal but fast)
        path = [start]
        current = list(start)
        goal = list(goal)
        
        # Simple path following
        while current != goal:
            if current[0] < goal[0]:
                current[0] += 1
            elif current[0] > goal[0]:
                current[0] -= 1
            
            if current[1] < goal[1]:
                current[1] += 1
            elif current[1] > goal[1]:
                current[1] -= 1
                
            path.append(tuple(current))
            
            # Prevent infinite loops
            if len(path) > grid_size * 2:
                break
        
        if len(path) > 1:
            paths.append(path)
    
    return paths

def generate_paths(cost_field, grid_size=50, num_paths=10, max_workers=None):
    """Generate optimal paths between random start and end points using parallel processing"""
    paths = []
    
    # Use ThreadPoolExecutor for parallel path generation
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all path generation tasks
        future_to_path = {
            executor.submit(generate_single_path, cost_field, grid_size): i 
            for i in range(num_paths * 2)  # Generate extra to account for failed paths
        }
        
        # Collect results as they complete with progress bar
        with tqdm(total=num_paths, desc="Generating paths") as pbar:
            for future in concurrent.futures.as_completed(future_to_path):
                path = future.result()
                if path is not None:
                    paths.append(path)
                    pbar.update(1)
                    if len(paths) >= num_paths:
                        break
    
    return paths[:num_paths]  # Return exactly the requested number

def visualize_results(true_cost_field, predicted_cost_field, paths):
    """Visualize the ground truth and predicted cost fields with paths"""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Create a colormap for the paths
    colors = plt.cm.tab10(np.linspace(0, 1, len(paths)))

    # Normalize both plots to the same scale
    vmin = min(true_cost_field.min(), predicted_cost_field.min())
    vmax = max(true_cost_field.max(), predicted_cost_field.max())

    # Ground truth cost field
    im1 = axes[0].imshow(true_cost_field, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0].set_title("Ground Truth Cost Field")
    for i, path in enumerate(paths):
        xs, ys = zip(*path)
        axes[0].plot(ys, xs, color=colors[i], linewidth=2.0, label=f'Path {i+1}')
    
    # Add colorbar for ground truth
    cbar1 = plt.colorbar(im1, ax=axes[0], shrink=0.8)
    cbar1.set_label('Cost Value')

    # Predicted cost field
    im2 = axes[1].imshow(predicted_cost_field, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[1].set_title("Predicted Cost Field")
    for i, path in enumerate(paths):
        xs, ys = zip(*path)
        axes[1].plot(ys, xs, color=colors[i], linewidth=2.0, label=f'Path {i+1}')
    
    # Add colorbar for predicted
    cbar2 = plt.colorbar(im2, ax=axes[1], shrink=0.8)
    cbar2.set_label('Cost Value')

    # Add legends with more space
    axes[0].legend(bbox_to_anchor=(1.15, 1), loc='upper left')
    axes[1].legend(bbox_to_anchor=(1.15, 1), loc='upper left')

    # Adjust layout to prevent overlap
    plt.subplots_adjust(right=0.85, wspace=0.3)
    plt.show()

def main():
    """Main execution function"""
    # Set random seed for reproducibility
    np.random.seed(42)

    # Grid setup
    grid_size = 50
    grid = np.zeros((grid_size, grid_size))

    # Define 3 items at fixed locations
    # Note: For now each item has a weight that is a constant which is then multiplied by a gaussian decay function
    items = [(10, 10), (40, 15), (25, 40)]
    true_weights = [5.0, 3.0, 7.0]  # Ground truth cost multipliers

    # Compute true cost field as maximum at each location
    cost_fields = [distance_cost_field(pos, w, grid_size) for pos, w in zip(items, true_weights)]
    true_cost_field = np.maximum.reduce(cost_fields)

    # Generate 1000 paths for better training
    print("Generating training paths...")
    # Use simple paths for faster training (comment out for optimal paths)
    use_simple_paths = False #True
    if use_simple_paths:
        paths = generate_simple_paths(true_cost_field, grid_size, num_paths=1000)
        print(f"Generated {len(paths)} simple paths")
    else:
        paths = generate_paths(true_cost_field, grid_size, num_paths=1000)
        print(f"Generated {len(paths)} optimal paths")

    # In a real scenario, we only have access to:
    # 1. Item locations
    # 2. Optimal paths (the paths that were chosen)
    # We do NOT have access to path costs or true parameters
    
    print("Learning cost function parameters from optimal paths only...")
    print(f"Available information: {len(items)} items, {len(paths)} optimal paths")

    # Train neural network to learn cost function parameters
    print("Training neural network to learn cost function parameters...")
    print(f"Training on {len(paths)} paths...")
    model = train_cost_function_model(paths, items, num_epochs=2000, lr=0.0001)
    
    # Extract learned parameters (average across all paths) - vectorized
    with torch.no_grad():
        # Get device from model
        device = next(model.parameters()).device
        
        # Process all paths at once for efficiency
        all_weights = []
        all_sigmas = []
        
        # Process in batches to avoid memory issues
        batch_size = 50
        with tqdm(total=len(paths), desc="Extracting parameters") as pbar:
            for i in range(0, len(paths), batch_size):
                batch_paths = paths[i:i+batch_size]
                
                for path in batch_paths:
                    # Vectorized distance computation
                    path_array = np.array(path)
                    items_array = np.array(items)
                    distances = np.linalg.norm(path_array[:, np.newaxis, :] - items_array[np.newaxis, :, :], axis=2)
                    
                    # Convert to tensor
                    distances_tensor = torch.tensor(distances, dtype=torch.float32, device=device)
                    
                    # Get predictions
                    encoded = model.feature_encoder(distances_tensor)
                    weights = model.weight_activation(model.weight_predictor(encoded))
                    sigmas = model.sigma_activation(model.sigma_predictor(encoded))
                    
                    # Average across path points
                    all_weights.append(weights.mean(dim=0))
                    all_sigmas.append(sigmas.mean(dim=0))
                    
                    pbar.update(1)
        
        # Average across all paths
        predicted_weights = torch.stack(all_weights).mean(dim=0).cpu().numpy()
        predicted_sigmas = torch.stack(all_sigmas).mean(dim=0).cpu().numpy()
    
    print(f"Learned weights: {predicted_weights}")
    print(f"Learned sigmas: {predicted_sigmas}")

    # Generate predicted cost field using learned weights and sigmas
    predicted_fields = [distance_cost_field(pos, w, grid_size, sigma) 
                       for pos, w, sigma in zip(items, predicted_weights, predicted_sigmas)]
    predicted_cost_field = np.maximum.reduce(predicted_fields)

    # Print comparison of weights and sigmas
    print("\nParameter Comparison:")
    print("True Weights:     ", true_weights)
    print("Predicted Weights:", predicted_weights)
    print("True Sigmas:      ", [5.0] * len(items))  # We used sigma=5.0 for all items
    print("Predicted Sigmas: ", predicted_sigmas)

    # Visualization (show only first 10 paths to avoid clutter)
    visualize_results(true_cost_field, predicted_cost_field, paths[:10])


if __name__ == "__main__":
    main()

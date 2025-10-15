import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import heapq

def dijkstra(cost, start, end):
    dist = np.full(cost.shape, np.inf)
    prev = np.empty(cost.shape + (2,), dtype=int)
    dist[start] = 0
    heap = [(0, start)]
    visited = set()
    while heap:
        d, (i, j) = heapq.heappop(heap)
        if (i, j) == end:
            break
        if (i, j) in visited:
            continue
        visited.add((i, j))
        for ni, nj in neighbors(i, j):
            alt = d + (cost[ni, nj] + cost[i, j]) / 2
            if alt < dist[ni, nj]:
                dist[ni, nj] = alt
                prev[ni, nj] = [i, j]
                heapq.heappush(heap, (alt, (ni, nj)))
    # Reconstruct path
    path = []
    curr = end
    while not np.array_equal(curr, start):
        path.append(curr)
        curr = tuple(prev[curr])
    path.append(start)
    path.reverse()
    return path

# Generate a synthetic scalar field φ(x, y) = hills
x = np.linspace(0, 10, 200)
y = np.linspace(0, 10, 200)
X, Y = np.meshgrid(x, y)

# Create a synthetic "terrain" φ(x, y)
phi = (
    3 * np.exp(-((X - 3)**2 + (Y - 3)**2) / 2) +
    2 * np.exp(-((X - 7)**2 + (Y - 5)**2) / 3) +
    1.5 * np.exp(-((X - 5)**2 + (Y - 8)**2) / 4)
)
phi = gaussian_filter(phi, sigma=2)  # Smooth terrain

# Dijkstra's algorithm for least-cost path
shape = phi.shape
cost = phi.copy()

# Helper to get neighbors
neighbors = lambda i, j: [
    (i+di, j+dj) for di, dj in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    if 0 <= i+di < shape[0] and 0 <= j+dj < shape[1]
]

# Choose multiple random start/end pairs
np.random.seed(42)
num_pairs = 5
pairs = []
for _ in range(num_pairs):
    start_idx = (np.random.randint(0, shape[0]), np.random.randint(0, shape[1]))
    end_idx = (np.random.randint(0, shape[0]), np.random.randint(0, shape[1]))
    pairs.append((start_idx, end_idx))

# Colors for each path
colors = ['yellow', 'magenta', 'cyan', 'lime', 'orange']

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Show all start and end points
for i, (start_idx, end_idx) in enumerate(pairs):
    axes[0].plot(x[start_idx[1]], y[start_idx[0]], marker='o', color=colors[i], label=f'Start {i+1}')
    axes[0].plot(x[end_idx[1]], y[end_idx[0]], marker='s', color=colors[i], label=f'End {i+1}')
axes[0].set_title("Start and End Points")
axes[0].set_xlim(0, 10)
axes[0].set_ylim(0, 10)
axes[0].set_aspect('equal')
axes[0].legend()

# Right: Paths overlaid on terrain
contour = axes[1].contourf(X, Y, phi, levels=50, cmap='terrain')
for i, (start_idx, end_idx) in enumerate(pairs):
    path = dijkstra(cost, start_idx, end_idx)
    path_x = [x[j] for ii, j in path]
    path_y = [y[ii] for ii, j in path]
    axes[1].plot(path_x, path_y, color=colors[i], linewidth=2, label=f'Path {i+1}')
    axes[1].plot(x[start_idx[1]], y[start_idx[0]], marker='o', color=colors[i])
    axes[1].plot(x[end_idx[1]], y[end_idx[0]], marker='s', color=colors[i])
axes[1].set_title("Least-Cost Paths Overlaid on Scalar Field (Terrain φ)")
axes[1].set_xlim(0, 10)
axes[1].set_ylim(0, 10)
axes[1].set_aspect('equal')
axes[1].legend()
fig.colorbar(contour, ax=axes[1], label='φ(x, y)')

plt.tight_layout()
plt.show()

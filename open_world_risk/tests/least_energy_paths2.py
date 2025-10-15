import numpy as np
import matplotlib.pyplot as plt

def compute_sign(o, p, v):
    a = o - p
    cross = a[0] * v[1] - a[1] * v[0]
    return np.sign(cross)

def compute_deflection(o, p, v, m, k):
    s = compute_sign(o, p, v)
    return k * s * m

def plot_geodesics(objects, m_values, geodesics, title, ax):
    sizes = (np.abs(m_values) * 80 + 40)
    colors = ['red' if m < 0 else 'blue' for m in m_values]
    ax.scatter(objects[:, 0], objects[:, 1], s=sizes, c=colors, edgecolor='k', alpha=0.8, label="Objects")
    for i, (obj, m) in enumerate(zip(objects, m_values)):
        ax.annotate(f"m={m:.2f}", (obj[0], obj[1]), textcoords="offset points", xytext=(5, 5), ha='left', fontsize=10)
    path_colors = ['#1b9e77', '#d95f02', '#7570b3']
    for j, geo in enumerate(geodesics):
        p = geo['p']
        v = geo['v']
        t_vals = np.linspace(-15, 15, 100)
        x_orig = p[0] + t_vals * v[0]
        y_orig = p[1] + t_vals * v[1]
        ax.plot(x_orig, y_orig, '--', color='gray', alpha=0.7, linewidth=1.5)
        x_def, y_def = [], []
        for t in t_vals:
            pos = p + t * v
            deflection = 0
            for i, (obj, m) in enumerate(zip(objects, m_values)):
                deflection += compute_deflection(obj, p, v, m, k)
            perp = np.array([-v[1], v[0]])
            deflected_pos = pos + deflection * perp
            x_def.append(deflected_pos[0])
            y_def.append(deflected_pos[1])
        ax.plot(x_def, y_def, '-', color=path_colors[j], linewidth=2, label=f'Geodesic {j+1}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)
    ax.set_xlim(-15, 5)
    ax.set_ylim(-2, 4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.15)
    ax.set_aspect('equal')
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)

def main():
    # Example setup
    # Object positions (2 objects, vertically separated)
    objects = np.array([[0, 0], [0, 2]])
    # True bending strengths (anti-gravity, negative)
    true_m = np.array([-1.0, -0.5])
    # Define geodesics: each as point p and direction v (unit vector)
    # We'll define 3 horizontal geodesics for simplicity
    geodesics = [
        {'p': np.array([-10, 3]), 'v': np.array([1, 0]) / np.linalg.norm([1, 0])},  # above both
        {'p': np.array([-10, 1]), 'v': np.array([1, 0]) / np.linalg.norm([1, 0])},  # between
        {'p': np.array([-10, -1]), 'v': np.array([1, 0]) / np.linalg.norm([1, 0])}, # below both
    ]
    # Constant k (set to 1 for simplicity; could be pi or 2*pi in a real model)
    k = 1.0
    # Compute matrix A and delta (simulated observations)
    N = len(geodesics)
    M = len(objects)
    A = np.zeros((N, M))
    delta = np.zeros(N)
    for j, geo in enumerate(geodesics):
        p = geo['p']
        v = geo['v']
        for i in range(M):
            o = objects[i]
            s = compute_sign(o, p, v)
            A[j, i] = k * s
        # Simulate observed deflection based on true_m
        delta[j] = np.dot(A[j, :], true_m)
    # Solve for m: least squares via pseudoinverse
    m_est = np.linalg.pinv(A) @ delta
    # Output
    print("True m:", true_m)
    print("A matrix:\n", A)
    print("Observed delta:", delta)
    print("Estimated m:", m_est)
    # Create visualizations
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    plot_geodesics(objects, true_m, geodesics, "Ground Truth", ax1)
    plot_geodesics(objects, m_est, geodesics, "Predicted", ax2)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()

if __name__ == '__main__':
    main()
#%%
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline

# Example data for two sets (simulated here)
x_data = np.linspace(0, 5, 100)
# y_data_1 = x_data*2 -5 + np.random.normal(scale=1, size=x_data.shape)
# y_data_2 = x_data*1.9 -6 + np.random.normal(scale=1, size=x_data.shape)

rand_inds1 = np.random.randint(0, 100, 20)
rand_inds2 = np.random.randint(0, 100, 20)
rand_inds3 = np.random.randint(0, 100, 20)

# Define B-spline basis
degree = 3
num_knots = 3
knots = np.linspace(0, 10, num_knots)
knots = np.concatenate(([0] * degree, knots, [10] * degree))  # Extended knots

# Design matrix for B-splines
spl_bases = [BSpline(knots, np.eye(len(knots) - degree - 1), degree)
             for _ in range(len(knots) - degree - 1)]
X = np.column_stack([bs(x_data) for bs in spl_bases])

# Variables for regression coefficients
beta_1 = cp.Variable(X.shape[1])
beta_2 = cp.Variable(X.shape[1])
beta_3 = cp.Variable(X.shape[1])

# Monotonicity constraints for both splines
constraints = [beta_1[i+1] <= beta_1[i] for i in range(X.shape[1]-1)]
constraints += [beta_2[i+1] <= beta_2[i] for i in range(X.shape[1]-1)]
constraints += [beta_3[i+1] <= beta_3[i] for i in range(X.shape[1]-1)]

# Difference constraint for ensuring S1 >= S2
# constraints += [beta_1[i] >= beta_2[i] for i in range(X.shape[1])]
# constraints += [beta_2[i] >= beta_3[i] for i in range(X.shape[1])]

# Enforce non-negativity of coefficients and less than 1
constraints += [(X @ beta_1)[0] == 1, (X @ beta_2)[0] == 1, (X @ beta_3)[0] == 1, 
               beta_1[-1] >= 0, beta_2[-1] >= 0, beta_3[-1] >= 0]

# Objective function for fitting both sets of data
# objective = cp.Minimize(cp.norm(X @ beta_1 - y_data_1, 2) + cp.norm(X @ beta_2 - y_data_2, 2))

# Maximize log likelihood of the data
objective = cp.Minimize( cp.sum( cp.maximum( (X @ beta_1)[rand_inds1], (X @ beta_2)[rand_inds2], (X @ beta_3)[rand_inds3] ) )
                         + 1*cp.sum_squares(beta_1[:-1] - beta_1[1:]) + 1*cp.sum_squares(beta_2[:-1] - beta_2[1:]) + 1*cp.sum_squares(beta_3[:-1] - beta_3[1:]))

# Optimization problem
problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.CLARABEL)

# Optimal coefficients
beta_1_opt = beta_1.value
beta_2_opt = beta_2.value
beta_3_opt = beta_3.value

# Construct the fitted splines
fitted_spline_1 = X @ beta_1_opt
fitted_spline_2 = X @ beta_2_opt
fitted_spline_3 = X @ beta_3_opt

# Plot results
# plt.scatter(x_data, y_data_1, label='Data 1', color='blue')
# plt.scatter(x_data, y_data_2, label='Data 2', color='green')
plt.plot(x_data, fitted_spline_1, label='Monotone Spline 1', color='red')
plt.plot(x_data, fitted_spline_2, label='Monotone Spline 2', color='orange')
plt.plot(x_data, fitted_spline_3, label='Monotone Spline 3', color='purple')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Monotone Polynomial Regression using B-splines with Positive Difference Constraint')

# Restrict axis range between 0 and 1 for y, 0 to 10 for x
plt.xlim(0, 5)
plt.ylim(0, 1)
plt.show()
# %%

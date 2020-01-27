from SALib.sample import saltelli, fast_sampler
from SALib.analyze import sobol
import numpy as np, matplotlib.pyplot as plt, matplotlib.font_manager as fm, os
from scipy.integrate import odeint
from mpl_toolkits.mplot3d.axes3d import Axes3D
from SALib.analyze import fast
from SALib.plotting.bar import plot as barplot
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


def lorenz(x, y, z, r, s=10, b=2.667):
    '''
    Given:
       x, y, z: a point of interest in three dimensional space
       s, r, b: parameters defining the lorenz attractor
    Returns:
       x_dot, y_dot, z_dot: values of the lorenz attractor's partial
           derivatives at the point x, y, z
    '''
    x_dot = s * (y - x)
    y_dot = r * x - y - x * z
    z_dot = x * y - b * z
    return x_dot, y_dot, z_dot


def distance(x, y):
    return np.sqrt(np.sum(np.square(x - y)))


dt = 0.01
num_steps = 10000

# Need one more for the initial values
xs = np.empty(num_steps + 1)
ys = np.empty(num_steps + 1)
zs = np.empty(num_steps + 1)

# Set initial values

xs[0], ys[0], zs[0] = (0, 1, 0)

# plt.show()
x_dot, y_dot, z_dot = lorenz(0, 1, 0, 24.08)
start_point = np.array((x_dot, y_dot, z_dot))

problem = {
    "num_vars": 4,
    "names": ['X', 'y', 'z', 'r'],
    "bounds": [[-10, 10], [-10, 10], [-10, 10], [1, 99]]
}

param_values = saltelli.sample(problem, num_steps)
Points = np.zeros([param_values.shape[0]])

for i in range(len(param_values)):
    x = param_values[i]
    x_dot, y_dot, z_dot = lorenz(x[0], x[1], x[2], x[3])
    a = np.array((x_dot, y_dot, z_dot))
    Points[i] = distance(a, start_point)

# print(Points.shape)
sensitivity = sobol.analyze(problem, Points)
print("Sobol Sensitivity S1:")
print(sensitivity["S1"])

print("Sobol Sensitivity ST:")
print(sensitivity["ST"])


# Fast analysis

fast_param_values = fast_sampler.sample(problem, num_steps)

FastPoints = np.zeros([fast_param_values.shape[0]])

for i in range(len(fast_param_values)):
    x = fast_param_values[i]
    x_dot, y_dot, z_dot = lorenz(x[0], x[1], x[2], x[3])
    a = np.array((x_dot, y_dot, z_dot))
    FastPoints[i] = distance(a, start_point)


fast_sensitivity = fast.analyze(problem, FastPoints, print_to_console=True)


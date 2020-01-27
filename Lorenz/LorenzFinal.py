from SALib.sample import saltelli, fast_sampler
from SALib.analyze import sobol, fast, morris
from SALib.sample.morris import sample
import numpy as np, matplotlib.pyplot as plt, matplotlib.font_manager as fm, os
from SALib.plotting.bar import plot as barplot
from scipy.integrate import odeint
from mpl_toolkits.mplot3d.axes3d import Axes3D
from SALib.test_functions import Ishigami
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
num_steps = 1000

# Need one more for the initial values
xs = np.empty(num_steps + 1)
ys = np.empty(num_steps + 1)
zs = np.empty(num_steps + 1)

# Set initial values
xs[0], ys[0], zs[0] = (0., 1., 0.)

Points = np.empty([num_steps, 3])

for i in range(num_steps):
    x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i], 24.08)
    xs[i + 1] = xs[i] + (x_dot * dt)
    ys[i + 1] = ys[i] + (y_dot * dt)
    zs[i + 1] = zs[i] + (z_dot * dt)
    a = np.array((x_dot, y_dot, z_dot))
    Points[i] = a

# Plot
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(xs, ys, zs, lw=0.5)
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.set_title("Lorenz Attractor")

plt.show()

problem = {
    "num_vars": 4,
    "names": ['X', 'y', 'z', 'r'],
    "bounds": [[-1, 1], [-1, 1], [-1, 1], [1, 50]]
}

sobol_param_values = saltelli.sample(problem, 100)
SobolPoints = []

fast_param_values = fast_sampler.sample(problem, 100)
FastPoints = []

morris_param_values = sample(problem, num_steps, num_levels=4)
MorrisPoints = []

for j in range(len(sobol_param_values)):
    x = sobol_param_values[j]
    xs = np.empty(num_steps + 1)
    ys = np.empty(num_steps + 1)
    zs = np.empty(num_steps + 1)
    xs[0], ys[0], zs[0] = (x[0], x[1], x[2])
    for i in range(num_steps):
        x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i], x[3])
        xs[i + 1] = xs[i] + (x_dot * dt)
        ys[i + 1] = ys[i] + (y_dot * dt)
        zs[i + 1] = zs[i] + (z_dot * dt)
        a = distance(np.array((x_dot, y_dot, z_dot)), Points[i])
        SobolPoints.append(a)

SobolArray = np.asarray(SobolPoints)
sensitivity = sobol.analyze(problem, SobolArray)
print("Sobol Sensitivity S1:")
print(sensitivity["S1"])
print("Sobol Sensitivity ST:")
print(sensitivity["ST"])

sensitivity.plot()


for j in range(len(fast_param_values)):
    x = fast_param_values[j]
    xs = np.empty(num_steps + 1)
    ys = np.empty(num_steps + 1)
    zs = np.empty(num_steps + 1)
    xs[0], ys[0], zs[0] = (x[0], x[1], x[2])
    for i in range(num_steps):
        x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i], x[3])
        xs[i + 1] = xs[i] + (x_dot * dt)
        ys[i + 1] = ys[i] + (y_dot * dt)
        zs[i + 1] = zs[i] + (z_dot * dt)
        a = distance(np.array((x_dot, y_dot, z_dot)), Points[i])
        FastPoints.append(a)

fast_sensitivity = fast.analyze(problem, np.asarray(FastPoints))
print("Fast Sensitivity Analysis: ")
print(fast_sensitivity)

# for j in range(morris_param_values):
#     x = morris_param_values[j]
#     xs = np.empty(num_steps + 1)
#     ys = np.empty(num_steps + 1)
#     zs = np.empty(num_steps + 1)
#     xs[0], ys[0], zs[0] = (x[0], x[1], x[2])
#     for i in range(num_steps):
#         x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i], x[3])
#         xs[i + 1] = xs[i] + (x_dot * dt)
#         ys[i + 1] = ys[i] + (y_dot * dt)
#         zs[i + 1] = zs[i] + (z_dot * dt)
#         a = distance(np.array((x_dot, y_dot, z_dot)), Points[i])
#         MorrisPoints.append(a)
#
# print('Morris Sensitivity:')
# morris_sensitivity = morris.analyze(problem, morris_param_values, np.asarray(MorrisPoints), conf_level=0.95,
#                                     print_to_console=True,
#                                     num_levels=4, num_resamples=100)

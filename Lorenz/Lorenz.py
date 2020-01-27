
from SALib.sample import saltelli
from SALib.analyze import sobol
import numpy as np, matplotlib.pyplot as plt, matplotlib.font_manager as fm, os
from scipy.integrate import odeint
from mpl_toolkits.mplot3d.axes3d import Axes3D
from SALib.test_functions import Ishigami
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


def lorenz(x, y, z, s=10, r=28, b=2.667):
    '''
    Given:
       x, y, z: a point of interest in three dimensional space
       s, r, b: parameters defining the lorenz attractor
    Returns:
       x_dot, y_dot, z_dot: values of the lorenz attractor's partial
           derivatives at the point x, y, z
    '''
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
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
start_point = np.array((0, 1, 0))
xs[0], ys[0], zs[0] = (0, 1, 0)

Points = np.empty(num_steps)

for i in range(num_steps):
    x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i])
    #calculate next point (x,y,z)
    xs[i + 1] = xs[i] + (x_dot * dt)
    ys[i + 1] = ys[i] + (y_dot * dt)
    zs[i + 1] = zs[i] + (z_dot * dt)
    # a = np.array((xs[i], ys[i], zs[i]))
    # calculate distance beetwen starting point and current point and add to
    a = np.array((x_dot, y_dot, z_dot))
    Points[i] = distance(a, start_point)



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
    "num_vars": 3,
    "names": ["x1", "x2", "x3"],
    "bounds": [[-50, 50], [-50, 50], [0, 50]]
}

sensitivity = sobol.analyze(problem, Points)
print("Sobol Sensitivity S1:")
print(sensitivity["S1"])

print("Sobol Sensitivity ST:")
print(sensitivity["ST"])
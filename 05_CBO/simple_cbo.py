import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import sys

sys.path.insert(0, os.getcwd())
from plot_results import get_plot_path

# Basic implementation of the CBO Algorithm described in "A consensus-based global
# optimization method for high dimensional machine learning problems" Section 2, Algorithm 2.1

def g(x):
    "Objective function"
    return (x[:,0]-3.14)**2 + (x[:,1]-2.72)**2 + np.sin(3*x[:,0]+1.41) + np.sin(4*x[:,1]-1.73)

def f(x):
    return (x[0]-1)**2 + (x[1]-5)**2

def rastrigin(x):
    A = 10
    return A * x.shape[1] + np.sum(x*x - A*np.cos(2*np.pi*x), 1)

# Step 0:
# Generate the random particles according to the same distribution rho_0 and
# introduce a remainder set R_0 to be empty
# We start with 2d example

dimensions = 2
root_particles = 10
n_particles = root_particles**2
beta = 20 # loss function weight, needs to be adjusted to typical values of problem
learning_rate = 0.05 # gamma or time step delta t
noise_rate = 0.1 # sigma
drift_rate = 1.0 # lambda
stopping_criterion = 1e-8
n_iterations = 100
particle_indices = list(range(0, n_particles))
noise = 'particle'

X = np.random.rand(n_particles, dimensions, ) * 10 - 5
x_range = (-5,5)
y_range = (-5,5)
# X = np.array(np.meshgrid(np.linspace(0,root_particles), np.linspace(0,5,5))).reshape(2,25).T

# Evaluate function to be minimised
L = rastrigin(X)
# Step 2.2 Calcualate the centroid of the set after the discrete analogue of Laplace's principle
mu = np.exp(-beta*L)
# Expand dimension of factor mu so elementwise multiplication with X is possible
mu = mu.reshape(-1, 1)
center = np.sum(X*mu, 0)/np.sum(mu, 0)

# Compute and plot the function in 3D within [0,5]x[0,5]
x, y = np.array(np.meshgrid(np.linspace(x_range[0], x_range[1], 100), np.linspace(y_range[0], y_range[1], 100)))
x_ = x.reshape(-1, 1)
y_ = y.reshape(-1, 1)
x_ = np.hstack((x_, y_))
z = rastrigin(x_).reshape(100, -1)

# Find the global minimum
# x_min = x.ravel()[z.argmin()]
# y_min = y.ravel()[z.argmin()]
x_min = 0.0
y_min = 0.0


# Set up base figure: The contour map
fig, ax = plt.subplots(figsize=(8,6))
fig.set_tight_layout(True)
img = ax.imshow(z, extent=[x_range[0], x_range[1], y_range[0], y_range[1]], origin='lower', cmap='viridis', alpha=0.3)
fig.colorbar(img, ax=ax)
ax.plot([x_min], [y_min], marker='x', markersize=5, color="white")
contours = ax.contour(x, y, z, 10, colors='black', alpha=0.4)
ax.clabel(contours, inline=True, fontsize=8, fmt="%.0f")
# pbest_plot = ax.scatter(pbest[0], pbest[1], marker='o', color='black', alpha=0.5)
p_plot = ax.scatter(X[:,0], X[:,1], marker='o', color='black', alpha=0.5)
# p_arrow = ax.quiver(X[0], X[1], V[0], V[1], color='blue', width=0.005, angles='xy', scale_units='xy', scale=1)
center_plot = plt.scatter([center[0]], [center[1]], marker='*', s=100, color='red', alpha=0.4)
ax.set_xlim(x_range)
ax.set_ylim(y_range)
# fig.savefig('test.png')

centers = []

# for k in range(n_steps):
#     # Step 1.1 Calculate the function values (or approximated values) of the to be
#     # minimised function at the location of the particles
#     L = rastrigin(X[:, 0], X[:, 1])
#     # Step 2.2 Calcualate the centroid of the set after the discrete analogue of Laplace's principle
#     mu = np.exp(-beta*L)
#     # Expand dimension of factor mu so elementwise multiplication with X is possible
#     mu = mu.reshape(-1, 1)
#     center = np.sum(X*mu, 0)/np.sum(mu, 0)
#     # Step 2.3 Update particles
#     # Calculate deviation of particles from common center from center deviation
#     deviation_from_center = X - center
#     consensus_term = drift_rate*learning_rate*deviation_from_center
#     # Calculate deviation from common center with random disturbance
#     normal_disturbance = np.random.normal(0.0, 1.0, (dimensions))
#     diffusion_term = noise_rate*np.sqrt(learning_rate)*deviation_from_center*normal_disturbance
#     X = X - consensus_term - diffusion_term
#     centers.append(center)
    # if k > 0:
    #     center_difference = centers[-2] - centers[-1]
    #     # Step 3: Check the stopping criterion
    #     if 1/dimensions*np.linalg.norm(center_difference)**2 <= stopping_criterion:
    #         break
    # if k%10==0:
    #     ax.scatter(center[0], center[1], color='red', alpha = k/(2*n_steps) + 0.5)
    #     p_plot = ax.scatter(X[:,0], X[:,1], marker='o', color='black', alpha=k/n_steps*0.1)


def update():

    global X, beta, drift_rate, learning_rate, noise_rate, centers, noise
    L = rastrigin(X)
    # Step 2.2 Calcualate the centroid of the set after the discrete analogue of Laplace's principle
    mu = np.exp(-beta*L)
    # Expand dimension of factor mu so elementwise multiplication with X is possible
    mu = mu.reshape(-1, 1)
    center = np.sum(X*mu, 0)/np.sum(mu, 0)
    # Step 2.3 Update particles
    # Calculate deviation of particles from common center from center deviation
    deviation_from_center = X - center
    consensus_term = drift_rate*learning_rate*deviation_from_center
    # Calculate deviation from common center with random disturbance
    if noise == 'particle':
        normal_disturbance = np.random.normal(0.0, 1.0, (X.shape[0], X.shape[1]))
        diffusion_term = noise_rate*np.sqrt(learning_rate)*deviation_from_center*normal_disturbance
    else:
        normal_disturbance = np.random.normal(0.0, 1.0, (X.shape[1]))
        diffusion_term = noise_rate*np.sqrt(learning_rate)*deviation_from_center*normal_disturbance
    X = X - consensus_term - diffusion_term
    centers.append(center)

def animate(i):
    "Steps of CBO. algorithm and show in plot"
    title = f'Iteration {i:02d}'
    global beta
    beta = beta
    update()
    ax.set_title(title)
    p_plot.set_offsets(X)
    center_plot.set_offsets(centers[-1].reshape(1, -1))
    return ax, p_plot, center_plot

anim = FuncAnimation(fig, animate, frames=list(range(n_iterations)), interval=50, blit=False, repeat=True)
path = os.path.abspath(__file__)
file_path = get_plot_path(path)
anim.save(file_path + '.gif', dpi=120, writer='imagemagick')


# fig.savefig(file_path + '.png')

print(f'True minimum: {x_min}, {y_min}')
print(f'CBO minimum: {centers[-1]}')
















# # Hyper-parameter of the algorithm
# # Memory coefficient
# c1 = 0.1
# # Social coefficient
# c2 = 0.2
# # Weight of current velocity
# w = 0.8

# # Create particles
# n_particles = 20
# np.random.seed(100)
# # Random starting positions
# X = np.random.rand(2, n_particles) * 5
# # Random starting velocity
# V = np.random.randn(2, n_particles) * 0.1

# # Initialize data
# pbest = X
# pbest_obj = f(X[0], X[1])
# gbest = pbest[:, pbest_obj.argmin()]
# gbest_obj = pbest_obj.min()

# def update():
#     "Function to do one iteration of particle swarm optimization"
#     global V, X, pbest, pbest_obj, gbest, gbest_obj
#     # Update params
#     r1, r2 = np.random.rand(2)
#     V = w * V + c1*r1*(pbest - X) + c2*r2*(gbest.reshape(-1,1)-X)
#     X = X + V
#     obj = f(X[0], X[1])
#     pbest[:, (pbest_obj >= obj)] = X[:, (pbest_obj >= obj)]
#     pbest_obj = np.array([pbest_obj, obj]).min(axis=0)
#     gbest = pbest[:, pbest_obj.argmin()]
#     gbest_obj = pbest_obj.min()

# # Set up base figure: The contour map
# fig, ax = plt.subplots(figsize=(8,6))
# fig.set_tight_layout(True)
# img = ax.imshow(z, extent=[0, 5, 0, 5], origin='lower', cmap='viridis', alpha=0.5)
# fig.colorbar(img, ax=ax)
# ax.plot([x_min], [y_min], marker='x', markersize=5, color="white")
# contours = ax.contour(x, y, z, 10, colors='black', alpha=0.4)
# ax.clabel(contours, inline=True, fontsize=8, fmt="%.0f")
# pbest_plot = ax.scatter(pbest[0], pbest[1], marker='o', color='black', alpha=0.5)
# p_plot = ax.scatter(X[0], X[1], marker='o', color='blue', alpha=0.5)
# p_arrow = ax.quiver(X[0], X[1], V[0], V[1], color='blue', width=0.005, angles='xy', scale_units='xy', scale=1)
# gbest_plot = plt.scatter([gbest[0]], [gbest[1]], marker='*', s=100, color='black', alpha=0.4)
# ax.set_xlim([0,5])
# ax.set_ylim([0,5])
# a = np.asarray(batch_centers_old)
# ax.scatter(a[:,0], a[:,1])
# fig.savefig('test.png')

# def animate(i):
#     "Steps of PSO: algorithm update and show in plot"
#     title = 'Iteration {:02d}'.format(i)
#     # Update params
#     update()
#     # Set picture
#     ax.set_title(title)
#     pbest_plot.set_offsets(pbest.T)
#     p_plot.set_offsets(X.T)
#     p_arrow.set_offsets(X.T)
#     p_arrow.set_UVC(V[0], V[1])
#     gbest_plot.set_offsets(gbest.reshape(1,-1))
#     return ax, pbest_plot, p_plot, p_arrow, gbest_plot

# anim = FuncAnimation(fig, animate, frames=list(range(1,50)), interval=500, blit=False, repeat=True)
# anim.save("PSO.gif", dpi=120, writer="imagemagick")

# print("PSO found best solution at f({})={}".format(gbest, gbest_obj))
# print("Global optimal at f({})={}".format([x_min,y_min], f(x_min,y_min)))
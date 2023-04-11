import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Basic implementation of the CBO Algorithm described in "A consensus-based global
# optimization method for high dimensional machine learning problems" Section 2, Algorithm 2.1

def f(x,y):
    "Objective function"
    return (x-3.14)**2 + (y-2.72)**2 + np.sin(3*x+1.41) + np.sin(4*y-1.73)

def g(x,y):
    return (x-4)**2 + (y-4)**2

# Step 0:
# Generate the random particles according to the same distribution rho_0 and
# introduce a remainder set R_0 to be empty
# We start with 2d example

dimensions = 2
n_particles = 25
batch_size = 10
beta_0 = 3.0
learning_rate_0 = 1.0# gamma
noise_rate_0 = 0.2 # sigma
drift_rate = 0.2 # lambda
stopping_criterion = 1e-6
particle_indices = list(range(0, n_particles))
X = np.random.rand(n_particles, dimensions) * 5
X = np.array(np.meshgrid(np.linspace(0,5,5), np.linspace(0,5,5))).reshape(2,25).T


remainder_set = []

n_steps = 50
batch_centers_old = []
batch_centers_new = []

# Compute and plot the function in 3D within [0,5]x[0,5]
x, y = np.array(np.meshgrid(np.linspace(0,5,100), np.linspace(0,5,100)))
z = g(x, y)

# Find the global minimum
x_min = x.ravel()[z.argmin()]
y_min = y.ravel()[z.argmin()]

# Set up base figure: The contour map
fig, ax = plt.subplots(figsize=(8,6))
fig.set_tight_layout(True)
img = ax.imshow(z, extent=[0, 5, 0, 5], origin='lower', cmap='viridis', alpha=0.1)
fig.colorbar(img, ax=ax)
ax.plot([x_min], [y_min], marker='x', markersize=5, color="white")
contours = ax.contour(x, y, z, 10, colors='black', alpha=0.4)
ax.clabel(contours, inline=True, fontsize=8, fmt="%.0f")
# pbest_plot = ax.scatter(pbest[0], pbest[1], marker='o', color='black', alpha=0.5)
p_plot = ax.scatter(X[:,0], X[:,1], marker='o', color='black', alpha=0.5)
# p_arrow = ax.quiver(X[0], X[1], V[0], V[1], color='blue', width=0.005, angles='xy', scale_units='xy', scale=1)
# gbest_plot = plt.scatter([gbest[0]], [gbest[1]], marker='*', s=100, color='black', alpha=0.4)
ax.set_xlim([0,5])
ax.set_ylim([0,5])
a = np.asarray(batch_centers_old)
# ax.scatter(a[:,0], a[:,1])
fig.savefig('test.png')

for k in range(n_steps):
    # Step 1 Choose Index batches
    permutation = list(np.random.permutation(particle_indices))
    index_list = remainder_set + permutation
    n_batches = np.int32(np.floor((n_particles + len(remainder_set))/batch_size))
    batches = [index_list[i*batch_size:(i+1)*batch_size] for i in range(n_batches)]
    remainder_set = index_list[n_batches*batch_size:]

    noise_rate = noise_rate_0/np.log(k+2)
    beta = beta_0*np.log(k+2)
    learning_rate = learning_rate_0*1/(k+1)

    # beta = beta_0
    # noise_rate = noise_rate_0
    # learning_rate = learning_rate_0




    for batch in batches:
        # Step 2.1 Calculate the function values (or approximated values) of the to be
        # minimised function at the location of the particles at the batches
        L = g(X[batch, 0], X[batch, 1])
        # Step 2.2 Calcualate center of batch particles
        mu = np.exp(-beta*L)
        # Expand dimension of factor mu so elementwise multiplication with X is possible
        mu = mu.reshape(-1, 1)
        batch_center = 1/np.sum(mu) * np.sum(X[batch]*mu, 0)
        # Step 2.3 Update particles
        # Calculate deviation of particles from common center from center deviation
        unscaled_deviation = X[batch]-batch_center
        center_deviation = drift_rate*learning_rate*unscaled_deviation
        # Calculate deviation from common center with random disturbance
        normal_disturbance = np.random.normal(0.0, 1.0, (batch_size, dimensions))
        disturbance_deviation = noise_rate*np.sqrt(learning_rate)*np.sum(unscaled_deviation*normal_disturbance, 1)
        X[batch] = X[batch] - center_deviation + disturbance_deviation.reshape(-1,1)
        batch_centers_new.append(batch_center)
    if k > 0:
        # the number of patch centers can vary from iteration to iteration, cull them
        # to the same length
        length_new = len(batch_centers_new)
        length_old = len(batch_centers_old)
        if length_new > length_old:
            n_batches = length_old
        else:
            n_batches = length_new
        batch_center_difference = np.array(batch_centers_new[:n_batches]) - np.array(batch_centers_old[:n_batches])
        # Step 3: Check the stopping criterion
        if 1/dimensions*np.linalg.norm(batch_center_difference)**2 <= stopping_criterion:
            break
    batch_centers_old = batch_centers_new.copy()
    batch_centers_new = []
    if k%10==0:
        a = np.asarray(batch_centers_old)
        ax.scatter(a[:,0], a[:,1], color='red', alpha = k/n_steps)
        p_plot = ax.scatter(X[:,0], X[:,1], marker='o', color='black', alpha=k/n_steps*0.1)
fig.savefig('test.png')
print(f'True minimum: {x_min}, {y_min}')
print(a)
















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
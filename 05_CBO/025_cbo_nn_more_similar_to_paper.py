import jax.numpy as jnp
import jax
from jax import random, disable_jit, flatten_util, jit
from flax import linen as nn
from flax.linen import initializers
from typing import Sequence, Tuple
import numpy as np
from functools import partial
from jax.config import config; config.update("jax_enable_x64", True)
import warnings
warnings.filterwarnings("error")

import matplotlib.pyplot as plt
from matplotlib import cm

import os
import sys

sys.path.insert(0, os.getcwd())
from plot_results import get_plot_path

def f(x):
    return (x)**2

# The Neural Network structure class
class ExplicitMLP(nn.Module):
    features: Sequence[int]
    def setup(self):
        # we automatically know what to do with lists, dicts of submodules
        self.layers = [nn.Dense(feat, bias_init=initializers.normal()) for feat in self.features]

        # for single submodules, we would just write:
        # self.layer1 = nn.Dense(feat1)

    def __call__(self, inputs):
        x = inputs
        for i, lyr in enumerate(self.layers):
            x = lyr(x)
            if i != len(self.layers) - 1:
                x = nn.relu(x)
        return x

    def set_key(self, seed=0):
        self.key = random.PRNGKey(0)

    def parameter_generator(self, sample_input):
        key1, key2 = random.split(self.key)
        self.key = key1
        parameters = self.init(key2, sample_input)
        flat_nn_parameters = self.flatten_parameters(parameters)
        return flat_nn_parameters

    def flatten_parameters(self, parameters):
        flat_nn_parameters, self.unravel_pytree = flatten_util.ravel_pytree(parameters)
        return flat_nn_parameters

    def deflatten_parameters(self, flat_parameters):
        return self.unravel_pytree(flat_parameters)




class CBO():
    beta: float
    gamma: float
    sigma: float
    lambda_: float
    n_particles: int
    n_iterations: int

    def __init__(self,
                 beta,
                 gamma,
                 sigma,
                 lambda_,
                 n_particles,
                 batch_size,
                 n_iterations):
        self.beta = beta
        self.gamma = gamma
        self.sigma = sigma
        self.lambda_ = lambda_
        self.n_particles = n_particles
        if batch_size < n_particles:
            self.batch_size = batch_size
        else:
            self.batch_size = n_particles
        self.n_iterations = n_iterations
        self.particle_indices = list(range(0, n_particles))

    def generate_particles(self, generator_function, *args):
        particles = []
        for i in range(self.n_particles):
            particles.append(generator_function(*args))
        self.particles = np.asarray(particles)

    def generate_batch(self, remainder_set):
        permutation = list(np.random.permutation(self.particle_indices))
        index_list = remainder_set + permutation
        n_batches = np.int32(np.floor((self.n_particles + len(remainder_set)) / self.batch_size))
        batches = [index_list[i * self.batch_size:(i + 1) * self.batch_size] for i in range(n_batches)]
        remainder_set = index_list[n_batches * self.batch_size:]
        self.n_batches = n_batches
        return batches, remainder_set

    def update(self,
               batch,
               evaluation_function,
               batch_centers,
               stopping_criterion):

        noise = 'xp'

        # batch_centers_new = []

        # for batch in batches:
        break_flag = False
        # Step 2.1 Calculate the function values (or approximated values) of the to be
        # minimised function at the location of the particles at the batches
        L = evaluation_function(parameters=self.particles[batch])
        n_particles = L.shape[0] # same as self.batch_size
        n_outputs = L.shape[1]
        n_parameters = self.particles.shape[1]
        # Step 2.2 Calcualate center of batch particles
        mu = np.exp(-self.beta * L)

        # Basically Matrix multiplication (commented out Line has identical result)
        # batch_center = self.particles[batch].T @ mu
        batch_center = np.einsum('xp,xo->op', self.particles[batch], mu)
        # Secondly normalise the center
        try:
            normalisation = 1/np.sum(mu, axis=0)
        except RuntimeWarning:
            # The mu values are so small that we devide by zero
            # replace them by small epsilon
            epsilon = 1e-8
            zero_like_indices = np.where(mu<epsilon)
            mu[zero_like_indices] = epsilon
            normalisation = 1/np.sum(mu, axis=0)
            # If this happens this usually means that beta is too large. Reduce beta
            self.beta = self.beta*0.99

        batch_center = np.einsum('op,o->op', batch_center, normalisation)
        # Step 2.3 Update particles
        # Calculate deviation of particles from common center from center deviation
        # Is Following For cascade replacable with EINSUM? Following EINSUM is NOT correct
        # deviation_from_center = np.einsum('xp,op->xop', self.particles[batch], -batch_center)
        deviation_from_center = np.empty((n_particles, n_outputs, n_parameters))

        # for x in range(n_particles):
        #     for o in range(n_outputs):
        #         deviation_from_center[x, o, :] = self.particles[batch][x, :] - batch_center[o, :]
        # Provided by ChatGPT3
        deviation_from_center[:, :, :] = self.particles[batch][:, np.newaxis, :] - batch_center[np.newaxis, :, :]


        consensus_term = self.lambda_ * self.gamma * deviation_from_center
        # Calculate deviation from common center with random disturbance
        if noise == 'xop':
            # Vary in all possible directions: each particle gets a variation for
            # each output of the evaluation function, and each parameter that gets optimized
            normal_disturbance = np.random.normal(0.0, 1.0, (n_particles, n_outputs, n_parameters))
            disturbance_term = self.sigma * np.sqrt(self.gamma) * deviation_from_center * normal_disturbance
        elif noise == 'xp':
            # Each particle has its own noise for each parameter, but the same for every
            # output
            normal_disturbance = np.random.normal(0.0, 1.0, (n_particles, n_parameters))
            disturbance_term = np.empty((n_particles, n_outputs, n_parameters))
            # for x in range(n_particles):
            #     for p in range(n_parameters):
            #         disturbance_term[x, :, p] = deviation_from_center[x, :, p] * normal_disturbance[x, p]
            # Provided by ChatGPT3
            disturbance_term = deviation_from_center * normal_disturbance[:, np.newaxis, :]

        elif noise == 'op':
            # Every particle gets the same noise, but it varies for each output and
            # for each parameter
            normal_disturbance = np.random.normal(0.0, 1.0, (n_outputs, n_parameters))
            disturbance_term = np.empty((n_particles, n_outputs, n_parameters))
            # for o in range(n_outputs):
            #     for p in range(n_parameters):
            #         disturbance_term[:, o, p] = deviation_from_center[:, o, p] * normal_disturbance[o, p]
            disturbance_term = deviation_from_center * normal_disturbance[np.newaxis, :, :]

        elif noise == 'p':
            # The noise only varies for the parameters. Each parameter has the same set
            # of noise and each output has also the same noise.
            normal_disturbance = np.random.normal(0.0, 1.0, (n_parameters))
            disturbance_term = self.sigma * np.sqrt(self.gamma) * deviation_from_center * normal_disturbance


        # Calculate mean for the consensus and disturbance team with respect to the outputs
        consensus_term = np.mean(consensus_term, axis=1)
        disturbance_term = np.mean(disturbance_term, axis=1)
        self.particles[batch] =  self.particles[batch] - consensus_term - disturbance_term

        if batch_centers != []:
            batch_center_difference = np.array(batch_center) - np.array(batch_centers[-1])
            # Step 3: Check the stopping criterion
            if 1/(self.n_batches*n_outputs*n_parameters) * np.linalg.norm(batch_center_difference)**2 <= stopping_criterion:
                self.gamma *= 1.01
                break_flag = True

        batch_centers.append(batch_center)

        return batch_centers, break_flag

def evaluation_function(training_set: Tuple, sample_index: np.array, model: ExplicitMLP, parameters):
    training_input = training_set[0]
    training_output = training_set[1]
    p = model.unravel_pytree(parameters)
    nn_output = model.apply(p, training_input[:,sample_index].T)
    loss_values = loss(nn_output, training_output[:,sample_index].T)
    loss_value = jnp.mean(loss_values, axis=0)
    return loss_value

def solution_function(training_set: Tuple, sample_index: np.array, model: ExplicitMLP, parameters):
    training_input = training_set[0]
    p = model.unravel_pytree(parameters)
    nn_output = model.apply(p, training_input[:,sample_index].T)
    return nn_output

def loss(x, x_ref):
    return (x-x_ref)**2

layers = [2,2,1]
neural_network = ExplicitMLP(features=layers)
neural_network.set_key()
# Choosing sample_batch_size such that it does not divide n_samples currently
# throws an error TODO
n_samples = 100
sample_batch_size = 50
sample_input_dimension = 1

# Random points in interval [-5, 5]
training_input = np.random.rand(sample_input_dimension, n_samples) * 10 - 5
# training_output = np.expand_dims(f(training_input),0)
training_output = f(training_input)
training_set = (training_input, training_output)


cbo = CBO(beta=5,
          gamma=1.0,
          sigma= 0.01,
          lambda_=1.0,
          n_particles=20,
          batch_size=10,
          n_iterations=1000)
sample_input = jnp.zeros((1,1))
cbo.generate_particles(neural_network.parameter_generator, sample_input)
test_parameters = neural_network.init(random.PRNGKey(99), sample_input)
sample_indices = np.asarray(list(range(n_samples)))
remainder_set = []
losses = []


# Plot the training data
# X = np.arange(-5,5,0.25)
# Y = np.arange(-5,5,0.25)
# XX, YY = np.meshgrid(X, Y)
# Z = f(np.asarray([XX, YY]))
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# surf = ax.plot_surface(XX, YY, Z, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=False, alpha=0.5)
# # Add a color bar which maps values to colors.
# fig.colorbar(surf, shrink=0.5, aspect=5)
# ax.scatter(training_input[0], training_input[1], training_output, s=2, c='k')
# fig.savefig('training_data_3D.png')

X = np.arange(-5,5,0.25)
Z = f(np.asarray(X))
fig, ax = plt.subplots()
ax.plot(X,Z)
ax.scatter(training_input, training_output)
fig.savefig('training_data_2D.png')

for i in range(cbo.n_iterations):

    cbo.gamma *= .99

    batch_centers = []

    # batched_sample_indices = np.split(sample_batch_indices, np.asarray(range(sample_batch_size, n_samples, sample_batch_size)), axis=0)
    particle_batches, remainder_set = cbo.generate_batch(remainder_set=remainder_set)
    # nudges = []

    for particle_batch in particle_batches:
        sample_batch_indices = np.random.permutation(sample_indices)[:sample_batch_size]
        partial_evaluation_function = jax.vmap(partial(evaluation_function, training_set=training_set, sample_index=sample_batch_indices, model=neural_network), 0)
        batch_centers, break_flag = \
            cbo.update(batch=particle_batch,
                       evaluation_function=partial_evaluation_function,
                       batch_centers=batch_centers,
                       stopping_criterion=1e-5)

    batch_center_losses = []

    # Each batch of particles has one center for each output of the target function
    # Calculate the performance of each center for all objectives. We assume each
    # objective is equally important and so just perform a simple mean to get a
    # performance evaluation of each center.
    batch_centers = np.asarray(batch_centers)
    for o in range(batch_centers.shape[1]):
        batch_center_loss = partial_evaluation_function(parameters=batch_centers[:, o,:])
        batch_center_losses.append(batch_center_loss)
    batch_center_losses = np.asarray(batch_center_losses).reshape(len(particle_batches), -1)
    batch_center_losses = batch_center_losses.mean(axis=1)
    batch_centers = batch_centers.mean(axis=1)
    # mean_loss = np.asarray(batch_center_losses).mean()

    # We use the performance of each center to calculate a new weighted center (of centers
    # so to say). Test this new center on a new batch of samples to get a evaluation
    # of the overall performance

    mu = np.exp(-cbo.beta * batch_center_losses)

    batch_center = np.einsum('xp,x->p', batch_centers, mu)
    # Secondly normalise the center
    try:
        normalisation = 1/np.sum(mu, axis=0)
    except RuntimeWarning:
        # The mu values are so small that we devide by zero
        # replace them by small epsilon
        epsilon = 1e-8
        zero_like_indices = np.where(mu<epsilon)
        mu[zero_like_indices] = epsilon
        normalisation = 1/np.sum(mu, axis=0)

    batch_center = batch_center * normalisation

    sample_batch_indices = np.random.permutation(sample_indices)[:sample_batch_size]
    overall_evaluation_function = jax.vmap(partial(evaluation_function, training_set=training_set, sample_index=sample_indices, model=neural_network), 0)
    overall_loss = overall_evaluation_function(parameters=batch_center.reshape(1, -1)).flatten()[0]

    print(f'Epoch: {i}, Loss: {overall_loss:.3f}, beta: {cbo.beta:.03f}, gamma: {cbo.gamma:.03f}')

    losses.append(overall_loss)

fig, ax = plt.subplots()
ax.plot(losses)
fig.savefig('losses.png')

nn_output = solution_function(training_set=training_set, sample_index=sample_indices, model=neural_network, parameters=batch_center)
fig, ax = plt.subplots()
ax.scatter(training_input.T, nn_output)
fig.savefig('predictions.png')




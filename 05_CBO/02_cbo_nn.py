import jax.numpy as jnp
from jax import random, disable_jit, flatten_util
from flax import linen as nn
from flax.linen import initializers
from typing import Sequence, Tuple
import numpy as np
from functools import partial
from jax.config import config; config.update("jax_enable_x64", True)
import warnings
warnings.filterwarnings("error")

import os
import sys

sys.path.insert(0, os.getcwd())
from plot_results import get_plot_path

def f(x):
    return (x[0]) + (x[1])

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
        self.batch_size = batch_size
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
        return batches, remainder_set

    def update(self,
               batches,
               evaluation_function,
               batch_centers_old,
               stopping_criterion):

        noise = 'all'

        batch_centers_new = []

        for batch in batches:
            break_flag = False
            # Step 2.1 Calculate the function values (or approximated values) of the to be
            # minimised function at the location of the particles at the batches
            L = evaluation_function(parameters=self.particles[batch])
            n_particles = L.shape[0] # same as self.batch_size
            n_samples = L.shape[1]
            n_outputs = L.shape[2]
            n_parameters = self.particles.shape[1]
            # Step 2.2 Calcualate center of batch particles
            mu = np.exp(-self.beta * L)
            # Expand dimension of factor mu so elementwise multiplication with X is possible
            # x = particle index
            # b = sample batch index
            # o = output index
            # p = parameter index
            # First weight every parameter of every particle by the result of the evaluation
            # function for every output and every sample in the sample batch and sum over
            # all particles to get the weighted center of all particles
            batch_center = np.einsum('xp,xbo->bop', self.particles[batch], mu)
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
            batch_center = np.einsum('bop,bo->bop', batch_center, normalisation)
            # Step 2.3 Update particles
            # Calculate deviation of particles from common center from center deviation
            # deviation_from_center = np.einsum('xp,bop->xbop', self.particles[batch], -batch_center)
            deviation_from_center = np.empty((n_particles, n_samples, n_outputs, n_parameters))
            for x in range(n_particles):
                for b in range(n_samples):
                    for o in range(n_outputs):
                        deviation_from_center[x, b, o, :] = self.particles[batch][x, :] - batch_center[b, o, :]
            # deviation_from_center = self.particles[batch] - batch_center
            consensus_term = self.lambda_ * self.gamma * deviation_from_center
            # Calculate deviation from common center with random disturbance
            if noise == 'all':
                # Vary in all possible directions: each particle gets a variation for
                # each training sample, output of the evaluation function, and each parameter that gets optimized
                normal_disturbance = np.random.normal(0.0, 1.0, (n_particles, n_samples, n_outputs, n_parameters))
            else:
                normal_disturbance = np.random.normal(0.0, 1.0, (n_particles, n_samples, n_outputs, n_parameters))

            disturbance_deviation = self.sigma * np.sqrt(self.gamma) * deviation_from_center * normal_disturbance
            # How this batch of samples wants to nudge the particles
            sample_batch_particle_nudges = np.einsum('xp,xbop->xp', self.particles[batch], - consensus_term - disturbance_deviation)
            batch_centers_new.append(batch_center)
        if batch_centers_old != []:
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
            if 1/(n_batches*n_outputs*n_parameters*n_samples) * np.linalg.norm(batch_center_difference)**2 <= stopping_criterion:
                break_flag = True

        batch_centers_old = batch_centers_new.copy()
        batch_centers_new = []
        return sample_batch_particle_nudges, batch_centers_old, break_flag

layers = [1]
neural_network = ExplicitMLP(features=layers)
neural_network.set_key()

cbo = CBO(beta=1.0,
          gamma=0.1,
          sigma= 0.01,
          lambda_=1.0,
          n_particles=10,
          batch_size=20,
          n_iterations=100)
sample_input = jnp.zeros((1,2))
cbo.generate_particles(neural_network.parameter_generator, sample_input)

n_samples = 100
# Choosing sample_batch_size such that it does not divide n_samples currently
# throws an error TODO
sample_batch_size = 1000
sample_input_dimension = 2

# Random points in interval [-5, 5]
training_input = np.random.rand(sample_input_dimension, n_samples) * 10 - 5
training_output = np.expand_dims(f(training_input),0)
training_set = (training_input, training_output)
test_parameters = neural_network.init(random.PRNGKey(99), sample_input)


def evaluation_function(training_set: Tuple, sample_index: np.array, model: ExplicitMLP, parameters: np.array):
    training_input = training_set[0]
    training_output = training_set[1]
    res = []
    for p in parameters:
        p = model.unravel_pytree(p)
        nn_output = model.apply(p, training_input[:,sample_index].T)
        res.append(loss(nn_output, training_output[:,sample_index].T))
    return np.asarray(res)

def loss(x, x_ref):
    return np.abs(x-x_ref)


sample_batch_indices = np.asarray(list(range(n_samples)))

remainder_set = []
for i in range(cbo.n_iterations):
    sample_lr = 0.01
    batch_centers_old = []
    sample_batch_indices = np.random.permutation(sample_batch_indices)
    batched_sample_indices = np.split(sample_batch_indices, np.asarray(range(sample_batch_size, n_samples, sample_batch_size)), axis=0)
    particle_batches, remainder_set = cbo.generate_batch(remainder_set=remainder_set)
    nudges = []
    for s_b in batched_sample_indices:
        partial_evaluation_function = partial(evaluation_function, training_set=training_set, sample_index=s_b, model=neural_network)
        sample_batch_particle_nudges, batch_centers_old, break_flag = \
            cbo.update(batches=particle_batches,
                       evaluation_function=partial_evaluation_function,
                       batch_centers_old=batch_centers_old,
                       stopping_criterion=1e-6)
        nudges.append(sample_batch_particle_nudges)

    batch_center_losses = []
    nudges = sample_lr * np.mean(np.asarray(nudges), axis=(0,1))
    print(f'Max nudge: {np.max(nudges)}')
    print(f'Min nudge: {np.min(nudges)}')
    cbo.particles += nudges

    # there is a center for each patch of particles (since each particle gets tested
    # on a batch of samples and each particle can have multiple outputs the shape
    # of the batch center should be: sample_batch_size x n_outputs x n_parameters)
    # we currently assume only one output dimension though (n_outputs=1)
    batch_centers_old = np.asarray(batch_centers_old)
    for b in range(batch_centers_old.shape[1]):
        for o in range(batch_centers_old.shape[2]):
            batch_center_loss = evaluation_function(training_set=training_set,
                                                    sample_index=sample_batch_indices,
                                                    model=neural_network,
                                                    parameters=batch_centers_old[:,b, o, :])
            batch_center_losses.append(batch_center_loss.mean())
    mean_loss = np.asarray(batch_center_losses).mean()
    print(mean_loss)
    # cbo.particles

    #     if break_flag:
    #         break
    # if break_flag:
    #     break




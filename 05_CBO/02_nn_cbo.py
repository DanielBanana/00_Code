import jax
from jax import random, flatten_util
from flax import linen as nn
from flax.linen import initializers
from typing import Sequence, Tuple
import numpy as np
from functools import partial
from jax.config import config; config.update("jax_enable_x64", True)
import warnings
import matplotlib.pyplot as plt
import os
import sys

# To use the plot_results file we need to add the uppermost folder to the PYTHONPATH
# Only Works if file gets called from 00_Code
sys.path.insert(0, os.getcwd())
from plot_results import plot_results, get_file_path

def f(x):
    """Function we want the Neural Network to approximate

    Parameters
    ----------
    x : int, float or numpy.array
        input

    Returns
    -------
    int, float or numpy.array
        output
    """
    return x*x

# The Neural Network structure class
class ExplicitMLP(nn.Module):
    features: Sequence[int]
    def setup(self):
        self.layers = [nn.Dense(feat, bias_init=initializers.normal()) for feat in self.features]

    def __call__(self, inputs):
        """Function which gets called when the object is called directly

        Parameters
        ----------
        inputs : array-like
            Input for the neural network

        Returns
        -------
        array-like
            Output of the neural network
        """
        x = inputs
        for i, lyr in enumerate(self.layers):
            x = lyr(x)
            if i != len(self.layers) - 1:
                x = nn.relu(x)
        return x

    def set_key(self, seed=0):
        """Set an internal key for the random initialisation of the parameters. For
        reproducibility purposes

        Parameters
        ----------
        seed : int, optional
            by default 0
        """
        self.key = random.PRNGKey(seed)

    def parameter_generator(self, sample_input):
        """Generates parameters for the neural network object.

        Parameters
        ----------
        sample_input : array-like
            The neural network needs a sample input to know which size of objects to expect.

        Returns
        -------
        array
            Flattend neural network parameters. Function to unflatten parameters: deflatten_parameters
        """
        key1, key2 = random.split(self.key)
        self.key = key1
        parameters = self.init(key2, sample_input)
        flat_nn_parameters = self.flatten_parameters(parameters)
        return flat_nn_parameters

    def flatten_parameters(self, parameters):
        """Receives unflattend parameters in form of a dict and flattens them to an array.

        Parameters
        ----------
        parameters : dict
            NN parameters used by the object.

        Returns
        -------
        array
            flattened parameters
        """
        flat_nn_parameters, self.unravel_pytree = flatten_util.ravel_pytree(parameters)
        return flat_nn_parameters

    def deflatten_parameters(self, flat_parameters):
        """Takes flattened parameters and returns them to the right dict from such that
        they are usable by the ExplicitMLP object.

        Parameters
        ----------
        flat_parameters : array

        Returns
        -------
        dict
            Unflattended NN parameters
        """
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
                 eps,
                 n_particles,
                 batch_size,
                 n_iterations):
        """Initialise Consensus-based optimisation object

        Parameters
        ----------
        beta : float
            Inverse temperature
        gamma : float
            learning rate for the particles
        sigma : float
            noise rate
        lambda_ : float
            drift rate
        n_particles : int
            Number of particles for the optimisation
        batch_size : int
            Number of particles per batch; does not need to split n_particles evenly
        n_iterations : int
            Number of optimisation iterations
        """
        self.beta = beta
        self.gamma = gamma
        self.sigma = sigma
        self.lambda_ = lambda_
        self.eps = eps
        self.n_particles = n_particles
        if batch_size < n_particles:
            self.batch_size = batch_size
        else:
            self.batch_size = n_particles
        self.n_iterations = n_iterations
        self.particle_indices = list(range(0, n_particles))

        self.particles = None
        self.batch = None
        self.consensus_point = None
        self.consensus_point_old = None
        self.epoch = 1

        self.training_input = None
        self.training_output = None


    def generate_particles(self, generator_function, *args):
        """Generates particles for the CBO via a generator function and its arguments

        Parameters
        ----------
        generator_function : function
            Function which returns an object which should be used as a particle for CBO
        """
        particles = []
        for i in range(self.n_particles):
            particles.append(generator_function(*args))
        self.particles = np.asarray(particles)

    def generate_batch(self, remainder_list):
        """Split the particles into batches. Considers the remainder set of last iteration
        while splitting. Returns also a new remainder set

        Parameters
        ----------
        remainder_list : List
            List of indices which were not packed into batches last iteration.

        Returns
        -------
        List of particle indices, List of remainder indices
        """
        permutation = list(np.random.permutation(self.particle_indices))
        index_list = remainder_list + permutation
        n_batches = np.int32(np.floor((self.n_particles + len(remainder_list)) / self.batch_size))
        batches = [index_list[i * self.batch_size:(i + 1) * self.batch_size] for i in range(n_batches)]
        remainder_list = index_list[n_batches * self.batch_size:]
        self.n_batches = n_batches
        return batches, remainder_list

    def maybe_random_shift(self, consensus_point_old, consensus_point):
        norm = np.linalg.norm(consensus_point_old-consensus_point,ord=float('inf'), axis=0)
        if np.less(norm, self.eps):
            self.particles += 10 * self.sigma * self.gamma**0.5 * np.random.normal(0.0, 1.0, (self.particles.shape[0], self.particles.shape[1]))

    def compute_consensus(self, batch=None, energy_values=None):
        batch = self.batch if batch is None else batch
        energy_values = self.compute_batch_energy_values(batch) if energy_values is None else energy_values
        weights = np.exp(-self.beta * (energy_values - energy_values.min()))
        batch_consensus = weights * self.particles[batch] / weights.sum()
        return batch_consensus.sum(axis=0), energy_values

    def compute_batch_energy_values(self, batch=None):
        batch = self.batch if batch is None else batch
        return self.evaluation_function(parameters=self.particles[self.batch])

    def compute_energy_values(self):
        return self.evaluation_function(parameters=self.particles)

    def update(self,
               batch,
               evaluation_function):
        """Performs one CBO step

        Parameters
        ----------
        batch : List of int
            List of indices for the particles of the current batch
        evaluation_function : function
            Function which takes a array/list of particles as input and returns their loss
        batch_consensuses : List
            List of center points of all batches in this iteration which have already
            been evaluated.
        stopping_criterion : float
            How close the batch center of this batch and the previous batch have to be
            to stop the optimisation process.

        Returns
        -------
        List
            List of batch_consensuses
        """
        noise = 'p'
        break_flag = False

        self.batch = batch
        self.evaluation_function = evaluation_function
        n_particles = len(batch)
        n_parameters = self.particles.shape[1]

        consensus_point, energy_values = self.compute_consensus(batch)

        # Step 2.3 Update particles
        # Calculate deviation of particles from common center from center deviation
        deviation_from_center = self.particles[batch] - consensus_point
        deviation_from_center = deviation_from_center

        consensus_term = self.lambda_ * self.gamma * deviation_from_center

        if noise == 'xp':
            # Each particle has its own noise for each parameter
            normal_disturbance = np.random.normal(0.0, 1.0, (n_particles, n_parameters))
            disturbance_term = self.sigma * np.sqrt(self.gamma) * deviation_from_center * normal_disturbance
        elif noise == 'p':
            # The noise only varies for the parameters. Each parameter has the same set
            # of noise
            normal_disturbance = np.random.normal(0.0, 1.0, (n_parameters))
            disturbance_term = self.sigma * np.sqrt(self.gamma) * deviation_from_center * normal_disturbance

        self.particles[batch] =  self.particles[batch] - consensus_term - disturbance_term

        return evaluation_function(parameters=consensus_point.reshape(1,-1))[0,0], consensus_point, break_flag

def evaluation_function(sample_set: Tuple, sample_index: np.array, model: ExplicitMLP, parameters):
    """The evaluation function for the update function of the CBO.

    Parameters
    ----------
    sample_set : Tuple
        Tuple of training inputs and training outputs
    sample_index : np.array
        Collection of indices which make up the batch of training samples for the current
        batch of particles.
    model : ExplicitMLP
        Object of an Neural Network implementation. Should have an apply() function
    parameters :
        Parameters for the model object

    Returns
    -------
    float
        Loss of the model with respect to the training batch
    """
    sample_input = sample_set[0]
    sample_output = sample_set[1]
    p = model.unravel_pytree(parameters)
    nn_output = model.apply(p, sample_input[:,sample_index].T)
    # loss_values = loss(nn_output, sample_output[:,sample_index].T)
    loss_values = (nn_output - sample_output[:,sample_index].T)**2
    loss_value = np.mean(loss_values, axis=0)
    return loss_value

def solution_function(sample_set: Tuple, sample_index: np.array, model: ExplicitMLP, parameters):
    """Evaluates the model for a batch of training samples without calculating the Loss.

    Parameters
    ----------
    sample_set : Tuple
        Tuple of inputs and outputs
    sample_index : np.array
        Collection of indices which make up the batch of training samples for the current
        batch of particles.
    model : ExplicitMLP
        Object of an Neural Network implementation. Should have an apply() function
    parameters :
        Parameters for the model object

    Returns
    -------
    array
        Output of the neural network for the parameters and the training batch.
    """
    sample_input = sample_set[0]
    p = model.unravel_pytree(parameters)
    nn_output = model.apply(p, sample_input[:,sample_index].T)
    return nn_output

def loss(x, x_ref):
    """Quadratic loss function

    Parameters
    ----------
    x : _type_
        _description_
    x_ref : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    return (x-x_ref)**2

path = os.path.abspath(__file__)
plot_path = get_file_path(path)

# NEURAL NETWORK
layers = [40, 1]
neural_network = ExplicitMLP(features=layers)
neural_network.set_key()
n_training_samples = 100
sample_batch_size = 20 # Choosing sample_batch_size such that it does not divide n_training_samples currently throws an error TODO
n_validation_samples = 20
sample_input_dimension = 1

# TRAINING AND VALIDATION DATA
# Random points in interval [-5, 5]
training_input = np.random.rand(sample_input_dimension, n_training_samples) * 10 - 5
validation_input = np.random.rand(sample_input_dimension, n_validation_samples) * 10 - 5
training_output = f(training_input)
validation_output = f(validation_input)
training_set = (training_input, training_output)
validation_set = (validation_input, validation_output)
training_sample_indices = np.asarray(list(range(n_training_samples)))
validation_sample_indices = np.asarray(list(range(n_validation_samples)))

# CBO
beta = 10
gamma = 0.1
sigma = 0.4 ** 0.5
lambda_ = 1.0
eps = 1e-5
n_particles = 1000
batch_size = 250
n_iterations = 2000
cooling = True
val_freq = 10
cbo = CBO(beta=beta,
          gamma=gamma,
          sigma=sigma,
          lambda_=lambda_,
          eps=eps,
          n_particles=n_particles,
          batch_size=batch_size,
          n_iterations=n_iterations)
sample_input = np.zeros((1,1))
cbo.generate_particles(neural_network.parameter_generator, sample_input)
test_parameters = neural_network.init(random.PRNGKey(42), sample_input)
test_parameters = neural_network.flatten_parameters(test_parameters)
cbo.model=neural_network


# TEST THE UNTRAINED NETWORK
nn_output = solution_function(sample_set=training_set, sample_index=training_sample_indices, model=neural_network, parameters=test_parameters.T)
fig, ax = plt.subplots()
X = np.arange(-5,5,0.25)
Z = f(np.asarray(X))
ax.plot(X,Z)
ax.scatter(training_input.T, nn_output)
fig.savefig(plot_path + '_' + 'untrained_predictions.png')

# PLOT THE TRAINING DATA
X = np.arange(-5,5,0.25)
Z = f(np.asarray(X))
fig, ax = plt.subplots()
ax.plot(X,Z)
ax.scatter(training_input, training_output)
fig.savefig(plot_path + '_' + 'training_data_2D.png')



remainder_list = []
train_losses = []
val_losses = []
for epoch in range(cbo.n_iterations):
    epoch_train_losses = []
    epoch_consensuses = []
    particle_batches, remainder_list = cbo.generate_batch(remainder_list=remainder_list)
    for batch_index, particle_batch in enumerate(particle_batches):
        sample_batch_indices = np.random.permutation(training_sample_indices)[:sample_batch_size]

        # We now create a partial function which we then parallelise with jax.vmap
        # The partial function is just a function which has some of its parameters pre set
        # It just results in a function which now takes fewer parameters
        # Vmap allows us to map the function for multiple inputs which speeds up the evaluation.
        # Partial gives us a function which now only takes different (neural network) parameters as input.
        # And vmap evaluates the partial function for a whole array of parameters
        partial_evaluation_function = jax.vmap(partial(evaluation_function, sample_set=training_set, sample_index=sample_batch_indices, model=neural_network), 0)
        training_loss, consensus_point, break_flag = cbo.update(batch=np.asarray(particle_batch),
                                evaluation_function=partial_evaluation_function)

        if epoch_consensuses != []:
            cbo.maybe_random_shift(epoch_consensuses[-1], consensus_point)
        epoch_train_losses.append(training_loss)
        epoch_consensuses.append(consensus_point)
        print(f'Epoch: {epoch}, batch: {(batch_index+1)*batch_size:4.0f}/{n_particles:4.0f}, training loss: {training_loss:3.3f}, beta: {cbo.beta:.03f}, gamma: {cbo.gamma:.03f}, sigma: {cbo.sigma:.08f}')

    if epoch % val_freq == 0 or epoch == cbo.n_iterations - 1:
        full_training_loss = val_loss = evaluation_function(sample_set=training_set, sample_index=range(len(training_set)), model=neural_network, parameters=epoch_consensuses[-1])
        val_loss = evaluation_function(sample_set=validation_set, sample_index=range(len(validation_set)), model=neural_network, parameters=epoch_consensuses[-1])
        print(f'Full training loss: {full_training_loss[0]:4.4f}, validation loss: {val_loss[0]:4.4f}')
    epoch_train_loss = np.asarray(epoch_train_losses).mean()
    val_losses.append(val_loss)
    train_losses.append(epoch_train_loss)
    if cooling:
        # cbo.sigma = cbo.sigma * np.log10(epoch + 2) / np.log10(epoch + 3)
        cbo.beta = cbo.beta * 1.001
        # cbo.gamma = cbo.gamma/(epoch+2) * (epoch+1)
        cbo.gamma = cbo.gamma * 0.999


# PLOT THE RESULTS
fig, ax = plt.subplots()
train_losses = np.asarray(train_losses)
ax.plot(train_losses)
fig.savefig(plot_path + '_' + 'train_losses.png')

fig, ax = plt.subplots()
val_losses = np.asarray(val_losses)
ax.plot(val_losses)
fig.savefig(plot_path + '_' + 'validation_losses.png')

nn_output = solution_function(sample_set=training_set, sample_index=training_sample_indices, model=neural_network, parameters=consensus_point.T)
fig, ax = plt.subplots()
X = np.arange(-5,5,0.25)
Z = f(np.asarray(X))
ax.plot(X,Z)
ax.scatter(training_input.T, nn_output)
fig.savefig(plot_path + '_' + 'train_predictions.png')

nn_output = solution_function(sample_set=validation_set, sample_index=validation_sample_indices, model=neural_network, parameters=consensus_point.T)
fig, ax = plt.subplots()
ax.plot(X,Z)
ax.scatter(validation_input.T, nn_output)
fig.savefig(plot_path + '_' + 'val_predictions.png')




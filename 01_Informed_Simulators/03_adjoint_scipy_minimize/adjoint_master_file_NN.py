import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize
import jax
from jax import random, jit, flatten_util, numpy as jnp
from flax import linen as nn
from flax.core import freeze, unfreeze
import orbax.checkpoint
from flax.training import orbax_utils
from typing import Sequence
import sys
import time
import argparse
from functools import partial
from jax import lax

# To use the plot_results file we need to add the uppermost folder to the PYTHONPATH
# Only Works if file gets called from 00_Code
sys.path.insert(0, os.getcwd())
from plot_results import plot_results, plot_losses, get_file_path

'''
Naming Conventions:
    z       refers to the state
    x       refers to the location variable of the state
    v       refers to the velocity variable of the state
    t       refers to time
    f       refers to the ode function
    g       refers to the inner part of the loss function: loss = sum(g) / loss = integral(g)
    d       refers to a total derivative
    del     refers to a partial derivative
    adj     refers to the adjoint state
    phi     collection of physical parameters (kappa, mu, mass)
    theta   collection of neural network parameters
    '''

# this only works on startup!
from jax.config import config
config.update("jax_enable_x64", True)


@jit
def ode(z, t, ode_parameters):
    '''Calculates the right hand side of the original ODE.'''
    kappa = ode_parameters[0]
    mu = ode_parameters[1]
    mass = ode_parameters[2]
    derivative = jnp.array([z[1],
                           -kappa*z[0]/mass + (mu*(1-z[0]**2)*z[1])/mass])
    return derivative

@jit
def ode_res(z, t, ode_parameters):
    '''Calculates the right hand side of the deficient ODE.'''
    kappa = ode_parameters[0]
    mu = ode_parameters[1]
    mass = ode_parameters[2]
    derivative = jnp.array([z[1],
                           -kappa*z[0]/mass])
    return derivative

@jit
def hybrid_ode(z, t, ode_parameters, nn_parameters):
    '''Calculates the right hand side of the hybrid ODE, where
    the damping term is replaced by the neural network'''
    kappa = ode_parameters[0]
    mu = ode_parameters[1]
    mass = ode_parameters[2]
    derivative = jnp.array([jnp.array((z[1],)),
                            jnp.array((-kappa*z[0]/mass,)) + jitted_neural_network(nn_parameters, z)]).flatten()
    return derivative

@jit
def ode_stim(z, t, ode_parameters):
    '''Calculates the right hand side of the original ODE.'''
    kappa = ode_parameters[0]
    mu = ode_parameters[1]
    mass = ode_parameters[2]
    derivative = jnp.array([z[1],
                           -kappa*z[0]/mass + (mu*(1-z[0]**2)*z[1])/mass + 1.2*jnp.cos(0.628*t)])
    return derivative

@jit
def ode_stim_res(z, t, ode_parameters):
    '''Calculates the right hand side of the original ODE.'''
    kappa = ode_parameters[0]
    mu = ode_parameters[1]
    mass = ode_parameters[2]
    derivative = jnp.array([z[1],
                           -kappa*z[0]/mass + 1.2*jnp.cos(0.628*t)])
    return derivative

@jit
def hybrid_ode_stim(z, t, ode_parameters, nn_parameters):
    '''Calculates the right hand side of the hybrid ODE, where
    the damping term is replaced by the neural network'''
    kappa = ode_parameters[0]
    mu = ode_parameters[1]
    mass = ode_parameters[2]
    derivative = jnp.array([jnp.array((z[1],)),
                            jnp.array((-kappa*z[0]/mass,)) + jitted_neural_network(nn_parameters, z) + jnp.array(1.2*jnp.cos(0.628*t))] ).flatten()
    return derivative

@jit
def ode_aug(z, t, ode_parameters):
    '''Calculates the right hand side of the original ODE.'''
    kappa = ode_parameters[0]
    mu = ode_parameters[1]
    mass = ode_parameters[2]
    derivative = [z[1],
                 -kappa*z[0]/mass + (mu*(1-z[0]**2)*z[1])/mass]
    derivative += [z[i] for i in range(2, z.shape[0])]

    return jnp.array(derivative)

@jit
def hybrid_ode_aug(z, t, ode_parameters, nn_parameters):
    '''Calculates the right hand side of the hybrid ODE, where
    the damping term is replaced by the neural network'''
    kappa = ode_parameters[0]
    mu = ode_parameters[1]
    mass = ode_parameters[2]
    derivative = [jnp.array((z[1],)),
                  jnp.array((-kappa*z[0]/mass,)) + jitted_neural_network(nn_parameters, z)]
    derivative += [[z[i]] for i in range(2, z.shape[0])]
    derivative = jnp.array(derivative).flatten()
    return derivative

@jit
def ode_simple(z, t, ode_parameters):
    '''Calculates the right hand side of the original ODE.'''
    kappa = ode_parameters[0]
    mu = ode_parameters[1]
    mass = ode_parameters[2]
    derivative = jnp.array([-0.1 * z[0] - 1 * z[1],
                            1 * z[0] - 0.1 * z[1]])
    return derivative

@jit
def hybrid_ode_simple(z, t, ode_parameters, nn_parameters):
    '''Calculates the right hand side of the hybrid ODE, where
    the damping term is replaced by the neural network'''
    kappa = ode_parameters[0]
    mu = ode_parameters[1]
    mass = ode_parameters[2]
    derivative = jnp.array([jnp.array((-0.1 * z[0] - 1 * z[1],)),
                            jitted_neural_network(nn_parameters, z)]).flatten()
    return derivative

@jit
def adjoint_f(adj, z, z_ref, t, ode_parameters, nn_parameters):
    '''Calculates the right hand side of the adjoint system.'''
    df_dz = jax.jacobian(hybrid_ode, argnums=0)(z, t, ode_parameters, nn_parameters)
    dg_dz = jax.grad(g, argnums=0)(z, z_ref, ode_parameters, nn_parameters)
    d_adj = - df_dz.T @ adj - dg_dz
    return d_adj


def g(z, z_ref, ode_parameters, nn_parameters):
    '''Calculates the inner part of the loss function.

    This function can either take individual floats for z
    and z_ref or whole numpy arrays'''
    return jnp.mean(0.5 * (z_ref - z)**2, axis = 0)

@jit
def J(z, z_ref, ode_parameters, nn_parameters):
    '''Calculates the complete loss of a trajectory w.r.t. a reference trajectory'''
    return jnp.mean(g(z, z_ref, ode_parameters, nn_parameters))

@jit
def J_residual(inputs, outputs, nn_parameters):
    def squared_error(input, output):
        pred = jitted_neural_network(nn_parameters, input)
        return (output-pred)**2
    return jnp.mean(jax.vmap(squared_error)(inputs, outputs), axis=0)[0]

def create_residuals(z_ref, t, ode_parameters):
    z_dot = (z_ref[1:] - z_ref[:-1])/(t[1:] - t[:-1]).reshape(-1,1)
    v_ode = jax.vmap(lambda z_ref, t, ode_parameters: ode_res(z_ref, t, ode_parameters), in_axes=(0, 0, None))
    residual = z_dot - v_ode(z_ref[:-1], t[:-1], ode_parameters)
    return residual

# @jit
# def euler_step(z, t, i, ode_parameters):
#     dt = t[i+1] - t[i]
#     return z[i] + dt * ode(z[i], t[i], ode_parameters)

def f_euler(z0, t, ode_parameters):
    '''Applies forward Euler to the original ODE and returns the trajectory'''
    z = jnp.zeros((t.shape[0], z0.shape[0]))
    z = z.at[0].set(z0)
    i = jnp.asarray(range(t.shape[0]))
    euler_body_func = partial(f_step, t=t, ode_parameters = ode_parameters)
    final, result = lax.scan(euler_body_func, z0, i)
    z = z.at[1:].set(result[:-1])
    # z[1:] = result[:-1]

    # for i in range(len(t)-1):
    #     dt = t[i+1] - t[i]
    #     z[i+1] = z[i] + dt * ode(z[i], t[i], ode_parameters)
    #     # z[i+1] = euler_step(z, t, i, ode_parameters)
    return z

def f_step(prev_z, i, t, ode_parameters):
    t = jnp.asarray(t)
    dt = t[i+1] - t[i]
    next_z = prev_z + dt * ode(prev_z, t[i], ode_parameters)
    return next_z, next_z

def hybrid_euler(z0, t, ode_parameters, nn_parameters):
    '''Applies forward Euler to the hybrid ODE and returns the trajectory'''
    z = jnp.zeros((t.shape[0], z0.shape[0]))
    z = z.at[0].set(z0)
    i = jnp.asarray(range(t.shape[0]))
    # We can replace the loop over the time by a lax.scan this is 3 times as fast: 0.32-0.26 -> 0.11-0.9
    euler_body_func = partial(hybrid_step, t=t, ode_parameters=ode_parameters, nn_parameters=nn_parameters)
    final, result = lax.scan(euler_body_func, z0, i)
    z = z.at[1:].set(result[:-1])
    # for i in range(len(t)-1):
    #     dt = t[i+1] - t[i]
    #     z[i+1] = z[i] + dt * hybrid_ode(z[i], t[i], ode_parameters, nn_parameters)
    return z

def hybrid_step(prev_z, i, t, ode_parameters, nn_parameters):
    t = jnp.asarray(t)
    dt = t[i+1] - t[i]
    next_z = prev_z + dt * hybrid_ode(prev_z, t[i], ode_parameters, nn_parameters)
    return next_z, next_z

def adj_euler(a0, z, z_ref, t, ode_parameters, nn_parameters):
    '''Applies forward Euler to the adjoint ODE and returns the trajectory'''
    a = jnp.zeros((t.shape[0], a0.shape[0]))
    a = a.at[0].set(a0)
    i = jnp.asarray(range(t.shape[0]))
    # We can replace the loop over the time by a lax.scan this is 3 times as fast: 0.32-0.26 -> 0.11-0.9
    euler_body_func = partial(adjoint_step, z=z, z_ref=z_ref, t=t, ode_parameters=ode_parameters,nn_parameters=nn_parameters)
    final, result = lax.scan(euler_body_func, a0, i)
    a = a.at[1:].set(result[:-1])

    # for i in range(len(t)-1):
    #     dt = t[i+1] - t[i]
    #     d_adj = adjoint_f(a[i], z[i], z_ref[i], t[i], ode_parameters, nn_parameters)
    #     a[i+1] = a[i] + dt * d_adj
    return a

def adjoint_step(prev_a, i, z, z_ref, t, ode_parameters, nn_parameters):
    t = jnp.asarray(t)
    z = jnp.asarray(z)
    z_ref = jnp.asarray(z_ref)
    dt = t[i+1]-t[i]
    next_a = prev_a + dt * adjoint_f(prev_a, z[i], z_ref[i], t[i], ode_parameters, nn_parameters)
    return next_a, next_a


# Based on https://www.mathworks.com/help/deeplearning/ug/dynamical-system-modeling-using-neural-ode.html#TrainNeuralODENetworkWithRungeKuttaODESolverExample-14
def create_mini_batch(n_times_per_obs, mini_batch_size, X, adjoint, t):

    n_timesteps = t.shape[0]
    # Create batches of trajectories
    if n_timesteps-n_times_per_obs == 0:
        s = np.array([0])
    else:
        s = np.random.choice(range(n_timesteps-n_times_per_obs), mini_batch_size)

    x0 = X[s, :]
    a0 = adjoint[s, :]
    targets = np.empty((mini_batch_size, n_times_per_obs, X.shape[1]))
    adjoints = np.empty((mini_batch_size, n_times_per_obs, adjoint.shape[1]))
    ts = []

    for i in range(mini_batch_size):
        targets[i, 0:n_times_per_obs, :] = X[s[i] + 0:(s[i] + n_times_per_obs), :]
        adjoints[i, 0:n_times_per_obs, :] = adjoint[s[i] + 0:(s[i] + n_times_per_obs), :]
        ts.append(t[s[i]:s[i]+n_times_per_obs])

    return x0, targets, a0, adjoints, ts

def model_loss(z0, ts, ode_parameters, nn_parameters, targets):
    zs = []
    losses = []
    # Compute Predictions
    for i, ic in enumerate(z0):
        z = hybrid_euler(ic, ts[i], ode_parameters, nn_parameters)
        losses.append(J(z, targets[i], ode_parameters, nn_parameters))
        zs.append(z)

    return zs, np.asarray(losses).mean()

# Vectorize the  jacobian df_dtheta for all time points
df_dtheta_function = lambda z, t, phi, theta: jax.jacobian(hybrid_ode, argnums=3)(z, t, phi, theta)
vectorized_df_dtheta_function = jit(jax.vmap(df_dtheta_function, in_axes=(0, 0, None, None)))
df_dtheta_function = jit(df_dtheta_function)

df_dt_function = lambda z, t, phi, theta: jax.jacobian(hybrid_ode, argnums=2)(z, t, phi, theta)
df_dt_function = jit(df_dt_function)

# Vectorize the  jacobian dg_dtheta for all time points
dg_dtheta_function = lambda z, z_ref, phi: jax.grad(g, argnums=3)(z, z_ref, phi)
vectorized_dg_dtheta_function = jit(jax.vmap(dg_dtheta_function, in_axes=(0, 0, None, None)))

# The Neural Network structure class
class ExplicitMLP(nn.Module):
    features: Sequence[int]
    def setup(self):
        self.layers = [nn.Dense(feat, kernel_init=nn.initializers.normal(0.1), bias_init=nn.initializers.normal(0.1)) for feat in self.features]

    # layers = []
    # for feat in self.features:
    #     layers.append(nn.Dense(feat))
    #     layers.append(nn.Dropout(0.2))
    # self.layers = layers

    def __call__(self, inputs):
        x = inputs
        for i, lyr in enumerate(self.layers):
            x = lyr(x)
            if i != len(self.layers) - 1:
                x = nn.silu(x)
        return x

def create_nn(layers, z0):
    key1, key2, = random.split(random.PRNGKey(np.random.randint(0,100)), 2)
    neural_network = ExplicitMLP(features=layers)
    nn_parameters = neural_network.init(key2, np.zeros((1, z0.shape[0])))
    jitted_neural_network = jax.jit(lambda p, x: neural_network.apply(p, x))
    return jitted_neural_network, nn_parameters

def residual_wrapper(flat_nn_parameters, optimizer_args):
    # Unpack the arguments

    start = time.time()

    t = optimizer_args['time']
    z0 = optimizer_args['initial_condition']
    z_ref = optimizer_args['reference_solution']
    z_val = optimizer_args['validation_solution']
    ode_parameters = optimizer_args['reference_ode_parameters']
    unravel_pytree = optimizer_args['unravel_function']
    epoch = optimizer_args['epoch']
    losses = optimizer_args['losses']
    batching = optimizer_args['batching']
    random_shift = optimizer_args['random_shift']
    checkpoint_interval = optimizer_args['checkpoint_interval']
    results_path = optimizer_args['results_path']
    best_loss = optimizer_args['best_loss']
    checkpoint_manager = optimizer_args['checkpoint_manager']


    # Get the parameters of the neural network out of the array structure into the
    # tree structure
    nn_parameters = unravel_pytree(flat_nn_parameters)
    # Save best parameters

    # in this case we only want the second part since the first part is completly known
    # and the neural network only works on the second part
    outputs = create_residuals(z_ref, t, ode_parameters)[:,1]
    inputs = z_ref[:-1]

    z = hybrid_euler(z0, t, ode_parameters, nn_parameters)

    # loss = J_residual(inputs, outputs, nn_parameters)
    res_loss = J_residual(inputs, outputs, nn_parameters)
    res_loss, gradient = jax.value_and_grad(J_residual, argnums=2)(inputs, outputs, nn_parameters)
    true_loss = J(z, z_ref, ode_parameters, nn_parameters)
    flat_gradient, _ = flatten_util.ravel_pytree(gradient)
    optimizer_args['epoch'] += 1
    optimizer_args['losses'].append(true_loss)
    if epoch > 0 and true_loss < best_loss:
        optimizer_args['saved_nn_parameters'] = nn_parameters
        optimizer_args['best_loss'] = true_loss
        save_args = orbax_utils.save_args_from_target(nn_parameters)
        checkpoint_manager.save(epoch, nn_parameters, save_kwargs={'save_args': save_args})

    if epoch % checkpoint_interval == 0:
        plot_results(t, z, z_ref, results_path+f'_epoch_{epoch}')
        # optimizer_args['saved_nn_parameters'] = nn_parameters


    end = time.time()

    print(f'Pretraining: Epoch: {epoch}, Residual Loss: {res_loss:.7f}, True Loss: {true_loss:.7f}, Time: {end-start:3.3f}')
    return res_loss, flat_gradient

def function_wrapper(flat_nn_parameters, optimizer_args):
    '''This is a function wrapper for the optimisation function. It returns the
    loss and the jacobian'''

    start = time.time()

    # Unpack the arguments
    t = optimizer_args['time']
    z0 = optimizer_args['initial_condition']
    z_ref = optimizer_args['reference_solution']
    z_val = optimizer_args['validation_solution']
    ode_parameters = optimizer_args['reference_ode_parameters']
    unravel_pytree = optimizer_args['unravel_function']
    epoch = optimizer_args['epoch']
    losses = optimizer_args['losses']
    batching = optimizer_args['batching']
    random_shift = optimizer_args['random_shift']
    checkpoint_interval = optimizer_args['checkpoint_interval']
    results_path = optimizer_args['results_path']
    best_loss = optimizer_args['best_loss']
    checkpoint_manager = optimizer_args['checkpoint_manager']

    # Get the parameters of the neural network out of the array structure into the
    # tree structure
    nn_parameters = unravel_pytree(flat_nn_parameters)

    z = hybrid_euler(z0, t, ode_parameters, nn_parameters) # 0.01-0.06s
    z = np.nan_to_num(z, nan=1e8)
    a0 = np.zeros(z0.shape)
    # df_dtheta_trajectory = vectorized_df_dtheta_function(z, t, ode_parameters, nn_parameters)
    adjoint = adj_euler(a0, np.flip(z, axis=0), np.flip(z_ref, axis=0), np.flip(t), ode_parameters, nn_parameters)
    adjoint = np.flip(adjoint, axis=0)

    if batching:
        z0s, targets, a0s, adjoints, ts = create_mini_batch(40, 200, z_ref, adjoint, t)
        zs, loss = model_loss(z0s, ts, ode_parameters, nn_parameters, targets)
        gradients = []
        for z_, adjoint_, t_ in zip(zs, adjoints, ts):
            # Calculate the gradient of the hybrid ode with respect to the nn_parameters
            df_dtheta_trajectory = vectorized_df_dtheta_function(z_, t_, ode_parameters, nn_parameters)
            # For loop probably not the fastest; Pytree probably better
            # Matrix multiplication of adjoint variable with jacobian
            df_dtheta_trajectory = unfreeze(df_dtheta_trajectory)
            for layer in df_dtheta_trajectory['params']:
                # Sum the matmul result over the entire time_span to get the final gradients
                df_dtheta_trajectory['params'][layer]['bias'] = np.einsum("iN,iNj->j", adjoint_, df_dtheta_trajectory['params'][layer]['bias'])
                df_dtheta_trajectory['params'][layer]['kernel'] = np.einsum("iN,iNjk->jk", adjoint_, df_dtheta_trajectory['params'][layer]['kernel'])
            df_dtheta = df_dtheta_trajectory
            dJ_dtheta = df_dtheta
            flat_dJ_dtheta, _ = flatten_util.ravel_pytree(dJ_dtheta)
            gradients.append(flat_dJ_dtheta)
        flat_dJ_dtheta = np.asarray(gradients).mean(0)
    else:

        # Calculate the gradient of the hybrid ode with respect to the nn_parameters
        df_dtheta_trajectory = vectorized_df_dtheta_function(z, t, ode_parameters, nn_parameters)
        # For loop probably not the fastest; Pytree probably better
        # Matrix multiplication of adjoint variable with jacobian
        df_dtheta_trajectory = unfreeze(df_dtheta_trajectory)
        for layer in df_dtheta_trajectory['params']:
            # Sum the matmul result over the entire time_span to get the final gradients
            df_dtheta_trajectory['params'][layer]['bias'] = np.einsum("iN,iNj->j", adjoint, df_dtheta_trajectory['params'][layer]['bias'])
            df_dtheta_trajectory['params'][layer]['kernel'] = np.einsum("iN,iNjk->jk", adjoint, df_dtheta_trajectory['params'][layer]['kernel'])
        df_dtheta = df_dtheta_trajectory
        dJ_dtheta = df_dtheta
        flat_dJ_dtheta, _ = flatten_util.ravel_pytree(dJ_dtheta)

    loss = J(z, z_ref, ode_parameters, nn_parameters)

    if random_shift:
        if np.abs(loss - losses[-1]) < 0.1:
            flat_dJ_dtheta += np.random.normal(0, np.linalg.norm(flat_dJ_dtheta,2), flat_dJ_dtheta.shape)

    if epoch % checkpoint_interval == 0:
        plot_results(t, z, z_ref, results_path+f'_epoch_{epoch}')
        # optimizer_args['saved_nn_parameters'] = nn_parameters

        # checkpoint_manager.save(epoch, nn_parameters, save_kwargs={'save_args': save_args})

    optimizer_args['epoch'] += 1
    optimizer_args['losses'].append(loss)

    if epoch > 0 and loss < best_loss:
        optimizer_args['saved_nn_parameters'] = nn_parameters
        optimizer_args['best_loss'] = loss
        save_args = orbax_utils.save_args_from_target(nn_parameters)
        checkpoint_manager.save(epoch, nn_parameters, save_kwargs={'save_args': save_args})

    end = time.time()

    print(f'Epoch: {epoch}, Loss: {loss:.7f}, Time: {end-start:3.3f}')
    return loss, flat_dJ_dtheta/(t.shape[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mu', type=float, default=8.53, help='damping value of the VdP damping term')
    parser.add_argument('--start', type=float, default=0.0, help='Start value of the ODE integration')
    parser.add_argument('--end', type=float, default=20.0, help='End value of the ODE integration')
    parser.add_argument('--nsteps', type=float, default=4001, help='How many integration steps to perform')

    parser.add_argument('--layers', type=int, default=4, help='Number of hidden layers')
    parser.add_argument('--layer_size', type=int, default=20, help='Number of neurons in a hidden layer')

    parser.add_argument('--aug_state', type=bool, default=False, help='Whether or not to use the augemented state for the ODE dynamics')
    parser.add_argument('--aug_dim', type=int, default=4, help='Number of augment dimensions')
    parser.add_argument('--random_shift', type=bool, default=False, help='Whether or not to shift the gradient of training stagnates')
    parser.add_argument('--batching', type=bool, default=False, help='whether or not to batch the training data')
    parser.add_argument('--batch_size', type=int, default=200, help='batch size (for samples-level batching)')
    parser.add_argument('--times_per_obs', type=int, default=50, help='Over how many time steps one sample goes during batching')

    parser.add_argument('--stimulate', type=bool, default=False, help='Whether or not to use the stimulated dynamics')
    parser.add_argument('--simple_problem', type=bool, default=False, help='Whether or not to use a simple damped oscillator instead of VdP')

    parser.add_argument('--method', type=str, default='BFGS', help='Which optimisation method to use')
    parser.add_argument('--tol', type=float, default=1e-12, help='Tolerance for the optimisation method')
    parser.add_argument('--transfer_learning', type=bool, default=True, help='Tolerance for the optimisation method')
    parser.add_argument('--res_steps', type=float, default=500, help='Number of steps for the Pretraining on the Residuals')

    parser.add_argument('--build_plot', required=False, default=True, action='store_true',
                        help='specify to build loss and accuracy plot')
    parser.add_argument('--results_name', required=False, type=str, default='demo',
                        help='name under which the results should be saved, like plots and such')
    parser.add_argument('--checkpoint_interval', required=False, type=int, default=100,
                        help='path to save the resulting plot')
    parser.add_argument('--restore', required=False, type=float, default=False,
                        help='restore previous parameters')

    parser.add_argument('--eval_freq', type=int, default=100, help='evaluate test accuracy every EVAL_FREQ '
                                                                   'samples-level batches')
    args = parser.parse_args()

    path = os.path.abspath(__file__)
    directory = os.path.sep.join(path.split(os.path.sep)[:-1])
    file_path = get_file_path(path)
    # results_path = file_path + f'_{args.results_name}'

    result_directory = os.path.join(directory, 'result')
    if not os.path.exists(result_directory):
        os.mkdir(result_directory)
    result_path = os.path.join(result_directory, args.results_name)

    residual_directory = os.path.join(directory, 'residual')
    if not os.path.exists(residual_directory):
        os.mkdir(residual_directory)
    residual_path = os.path.join(residual_directory, args.results_name)

    checkpoint_directory = os.path.join(directory, 'ckpt')
    if not os.path.exists(checkpoint_directory):
        os.mkdir(checkpoint_directory)

    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=5, create=True)
    checkpoint_manager = orbax.checkpoint.CheckpointManager(checkpoint_directory, orbax_checkpointer, options)

    if args.aug_state:
        ode = ode_aug
        hybrid_ode = hybrid_ode_aug
        z0 = [1.0, 0.0]
        z0 += [0.0 for i in range(args.aug_dim)]
        z0 = np.asarray(z0)
    elif args.stimulate:
        ode = ode_stim
        hybrid_ode = hybrid_ode_stim
        ode_res = ode_stim_res
        z0 = np.array([1.0, 0.0])
    elif args.simple_problem:
        ode = ode_simple
        hybrid_ode = hybrid_ode_simple
        z0 = np.array([2.0, 0.0])
    else:
        z0 = np.array([1.0, 0.0])

    t_ref = np.linspace(args.start, args.end, args.nsteps)
    t_val = np.linspace(args.end, (args.end-args.start) * 1.5, int(args.nsteps * 0.5))
    reference_ode_parameters = np.asarray([1.0, args.mu, 1.0])
    z_ref = f_euler(z0, t_ref, reference_ode_parameters)
    z0_val = z_ref[-1]
    z_val = f_euler(z0_val, t_val, reference_ode_parameters)

    layers = [args.layer_size]*args.layers
    layers.append(1)
    jitted_neural_network, nn_parameters = create_nn(layers, z0)

    if args.restore:
        # Restore previous parameters
        step = checkpoint_manager.latest_step()
        nn_parameters = checkpoint_manager.restore(step)

    flat_nn_parameters, unravel_pytree = flatten_util.ravel_pytree(nn_parameters)

    epoch = 0

    # Put all arguments the optimization needs into one array for the minimize function
    optimizer_args = {'time' : t_ref,
                      'initial_condition' : z0,
                      'reference_solution' : z_ref,
                      'validation_solution' : z_val,
                      'reference_ode_parameters' : reference_ode_parameters,
                      'unravel_function' : unravel_pytree,
                      'epoch' : epoch,
                      'losses' : [],
                      'batching' : args.batching,
                      'random_shift' : args.random_shift,
                      'checkpoint_interval' : args.checkpoint_interval,
                      'results_path' : residual_path,
                      'saved_nn_parameters' : nn_parameters,
                      'best_loss' : np.inf,
                      'checkpoint_manager' : checkpoint_manager}


    # Train on Residuals
    ####################################################################################
    # Optimisers: CG, BFGS, Newton-CG, L-BFGS-B, TNC, SLSQP, dogleg, trust-ncg, trust-krylov, trust-exact and trust-constr
    if args.transfer_learning:
        residual_result = minimize(residual_wrapper, flat_nn_parameters, method=args.method, jac=True, args=optimizer_args, options={'maxiter':args.res_steps})
        nn_parameters = optimizer_args['saved_nn_parameters']
        best_loss = optimizer_args['best_loss']
        print(f'Best Loss in Residual Training: {best_loss}')
        flat_nn_parameters, _ = flatten_util.ravel_pytree(nn_parameters)
        # Plot the result of the training on residuals
        z = hybrid_euler(z0, t_ref, reference_ode_parameters, nn_parameters)
        path = os.path.abspath(__file__)
        plot_path = get_file_path(path)
        plot_results(t_ref, z, z_ref, residual_path+'_best')
        optimizer_args['epoch'] = 0

    # Train on Trajectory
    ####################################################################################
    optimizer_args['results_path'] = result_path
    try:

        res = minimize(function_wrapper, flat_nn_parameters, method='BFGS', jac=True, args=optimizer_args, tol=args.tol)
        print(res)
    except KeyboardInterrupt:
        pass
    nn_parameters = optimizer_args['saved_nn_parameters']
    best_loss = optimizer_args['best_loss']
    print(f'Best Loss in Training: {best_loss}')

    z_training = hybrid_euler(z0, t_ref, reference_ode_parameters, nn_parameters)
    z_validation = hybrid_euler(z0_val, t_val, reference_ode_parameters, nn_parameters)

    path = os.path.abspath(__file__)
    plot_path = get_file_path(path)
    plot_results(t_ref, z_training, z_ref, result_path+'_best')
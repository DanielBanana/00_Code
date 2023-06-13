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
import datetime
import argparse
from functools import partial
from jax import lax
from pyDOE2 import fullfact, bbdesign
import warnings
import shutil
from collections import OrderedDict
import yaml
warnings.filterwarnings("error")

# To use the plot_results file we need to add the uppermost folder to the PYTHONPATH
# Only Works if file gets called from 00_Code
sys.path.insert(0, os.getcwd())
from plot_results import plot_results, plot_losses, get_file_path
PI = np.pi

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
                           -kappa*z[0]/mass + (mu*(1-z[0]**2)*z[1])/mass + 1.2*jnp.cos(jnp.pi/5*t)])
    return derivative

@jit
def ode_stim_res(z, t, ode_parameters):
    '''Calculates the right hand side of the original ODE.'''
    kappa = ode_parameters[0]
    mu = ode_parameters[1]
    mass = ode_parameters[2]
    derivative = jnp.array([z[1],
                           -kappa*z[0]/mass + 1.2*jnp.cos(jnp.pi/5*t)])
    return derivative

@jit
def hybrid_ode_stim(z, t, ode_parameters, nn_parameters):
    '''Calculates the right hand side of the hybrid ODE, where
    the damping term is replaced by the neural network'''
    kappa = ode_parameters[0]
    mu = ode_parameters[1]
    mass = ode_parameters[2]
    derivative = jnp.array([jnp.array((z[1],)),
                            jnp.array((-kappa*z[0]/mass,)) + jitted_neural_network(nn_parameters, z) + jnp.array(1.2*jnp.cos(jnp.pi/5*t))] ).flatten()
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
    # cond = np.linalg.cond(df_dz)
    # return d_adj, cond
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

def create_residual_references(z_ref, t, ode_parameters):
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

def adjoint_euler(a0, z, z_ref, t, ode_parameters, nn_parameters):
    '''Applies forward Euler to the adjoint ODE and returns the trajectory'''
    a = jnp.zeros((t.shape[0], a0.shape[0]))
    a = a.at[0].set(a0)
    i = jnp.asarray(range(t.shape[0]))

    def adjoint_step(prev_a, i, z, z_ref, t, ode_parameters, nn_parameters):
        t = jnp.asarray(t)
        z = jnp.asarray(z)
        z_ref = jnp.asarray(z_ref)
        dt = t[i+1]-t[i]
        next_a = prev_a + dt * adjoint_f(prev_a, z[i], z_ref[i], t[i], ode_parameters, nn_parameters)
        return next_a, next_a

    # We can replace the loop over the time by a lax.scan this is 3 times as fast: 0.32-0.26 -> 0.11-0.9
    euler_body_func = partial(adjoint_step, z=z, z_ref=z_ref, t=t, ode_parameters=ode_parameters,nn_parameters=nn_parameters)
    final, result = lax.scan(euler_body_func, a0, i)
    a = a.at[1:].set(result[:-1])

    # a = np.zeros((t.shape[0], a0.shape[0]))
    # a[0] = a0
    # for i in range(len(t)-1):
    #     dt = t[i+1] - t[i]
    #     d_adj, cond = adjoint_f(a[i], z[i], z_ref[i], t[i], ode_parameters, nn_parameters)
    #     print(cond)
    #     # d_adj = adjoint_f(a[i], z[i], z_ref[i], t[i], ode_parameters, nn_parameters)
    #     a[i+1] = a[i] + dt * d_adj

    return a


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

def create_clean_mini_batch(n_mini_batches, x_ref, t):
    n_timesteps = t.shape[0]
    # Create batches of trajectories
    mini_batch_size = int(n_timesteps/n_mini_batches)
    s = [mini_batch_size * i for i in range(n_mini_batches)]
    x0 = x_ref[s, :]
    targets = [x_ref[s[i]:s[i]+mini_batch_size] for i in range(n_mini_batches)]
    ts = [t[s[i]:s[i]+mini_batch_size] for i in range(n_mini_batches)]
    return x0, targets, ts

def model_loss(z0s, ts, ode_parameters, nn_parameters, targets):
    zs = []
    adjoints = []
    losses = []
    # Compute Predictions
    a0 = np.zeros(z0s[0].shape)
    for i, ic in enumerate(z0s):
        t = ts[i]
        z_ref = targets[i]
        z = hybrid_euler(ic, t, ode_parameters, nn_parameters)
        losses.append(J(z, z_ref, ode_parameters, nn_parameters))
        zs.append(z)
        # df_dtheta_trajectory = vectorized_df_dtheta_function(z, t, ode_parameters, nn_parameters)
        adjoint = adjoint_euler(a0, np.flip(z, axis=0), np.flip(z_ref, axis=0), np.flip(t), ode_parameters, nn_parameters)
        adjoint = np.flip(adjoint, axis=0)
        adjoints.append(adjoint)
    return zs, adjoints, np.asarray(losses).mean()

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
    z_ref_val = optimizer_args['validation_solution']
    ode_parameters = optimizer_args['reference_ode_parameters']
    unravel_pytree = optimizer_args['unravel_function']
    epoch = optimizer_args['epoch']
    losses = optimizer_args['losses']
    residual_losses = optimizer_args['residual_losses']
    batching = optimizer_args['batching']
    random_shift = optimizer_args['random_shift']
    checkpoint_interval = optimizer_args['checkpoint_interval']
    results_directory = optimizer_args['results_directory']
    best_loss = optimizer_args['best_loss']
    checkpoint_manager = optimizer_args['checkpoint_manager']
    lambda_ = optimizer_args['lambda']
    loss_cutoff = optimizer_args['loss_cutoff']
    residual_outputs = optimizer_args['residual_output']

    # Get the parameters of the neural network out of the array structure into the
    # tree structure
    nn_parameters = unravel_pytree(flat_nn_parameters)

    # in this case we only want the second part since the first part is completly known
    # and the neural network only works on the second part
    residual_inputs = z_ref[:-1]

    # Calculate the loss and gradient over the Residuals
    res_loss, gradient = jax.value_and_grad(J_residual, argnums=2)(residual_inputs, residual_outputs, nn_parameters)
    flat_gradient, _ = flatten_util.ravel_pytree(gradient)

    # Calculate the Loss over the trajectory for the training and validation data
    z = hybrid_euler(z0, t, ode_parameters, nn_parameters)
    loss = J(z, z_ref, ode_parameters, nn_parameters)
    z_val = hybrid_euler(z_ref[-1], t_val, ode_parameters, nn_parameters)
    val_loss = J(z_val, z_ref_val, ode_parameters, nn_parameters)

    # Plot the results in regular intervals
    if epoch % checkpoint_interval == 0:
        # Due to batching the length of z can change
        plot_results(t[:z.shape[0]], z, z_ref[:z.shape[0]], os.path.join(results_directory,f'epoch_{epoch}'))
        plot_results(t_val, z_val, z_ref_val, os.path.join(results_directory,f'val_epoch_{epoch}'))

    # Save the best parameters measured on the validation data
    if epoch > 0 and val_loss < best_loss:
        optimizer_args['saved_nn_parameters'] = nn_parameters
        optimizer_args['best_loss'] = val_loss
        save_args = orbax_utils.save_args_from_target(nn_parameters)
        checkpoint_manager.save(epoch, nn_parameters, save_kwargs={'save_args': save_args})

    # Regularise the optimizer with a punishment on the nn_parameters (bigger parameter values
    # equals more a higher loss punishment)
    if lambda_ == 0.0:
        L2_regularisation = 0.0
    else:
        L2_regularisation = lambda_/(2*(t.shape[0])) * np.linalg.norm(flat_nn_parameters, 2)**2

    optimizer_args['epoch'] += 1
    optimizer_args['losses'].append(loss)
    optimizer_args['val_losses'].append(val_loss)
    optimizer_args['residual_losses'].append(res_loss)
    end = time.time()
    print(f'Pretraining: Epoch: {epoch}, Residual Loss: {res_loss:.5E}, Loss: {loss:.5E}, Validation Loss: {val_loss:.5E} Time: {end-start:3.3f}')

    # If the loss on the validation trajectory passes a lower bound the result is good enough
    if val_loss < loss_cutoff:
        warnings.warn('Terminating Optimization: Required Loss reached')

    return res_loss + L2_regularisation, flat_gradient

def function_wrapper(flat_nn_parameters, optimizer_args):
    '''This is a function wrapper for the optimisation function. It returns the
    loss and the jacobian'''

    start = time.time()

    # Unpack the arguments
    t = optimizer_args['time']
    t_val = optimizer_args['val_time']
    z0 = optimizer_args['initial_condition']
    z_ref = optimizer_args['reference_solution']
    z_ref_val = optimizer_args['validation_solution']
    ode_parameters = optimizer_args['reference_ode_parameters']
    unravel_pytree = optimizer_args['unravel_function']
    epoch = optimizer_args['epoch']
    losses = optimizer_args['losses']
    batching = optimizer_args['batching']
    n_batches = optimizer_args['n_batches']
    batch_size = optimizer_args['batch_size']
    clean_batching = optimizer_args['clean_batching']
    n_clean_batches = optimizer_args['n_clean_batches']
    random_shift = optimizer_args['random_shift']
    checkpoint_interval = optimizer_args['checkpoint_interval']
    results_directory = optimizer_args['results_directory']
    best_loss = optimizer_args['best_loss']
    checkpoint_manager = optimizer_args['checkpoint_manager']
    lambda_ = optimizer_args['lambda']
    clean_batching = optimizer_args['clean_batching']
    loss_cutoff = optimizer_args['loss_cutoff']

    # Get the parameters of the neural network out of the array structure into the
    # tree structure
    nn_parameters = unravel_pytree(flat_nn_parameters)

    if batching:
        if clean_batching:
            z0s, targets, ts = create_clean_mini_batch(n_clean_batches, z_ref, t)
        else:
            z0s, targets, a0s, adjoints, ts = create_mini_batch(n_batches, batch_size, z_ref, adjoint, t)
        zs, adjoints, loss = model_loss(z0s, ts, ode_parameters, nn_parameters, targets)
        gradients = []
        for z_, adjoint_, t_ in zip(zs, adjoints, ts):
            # Calculate the gradient of the hybrid ode with respect to the nn_parameters
            df_dtheta_trajectory = vectorized_df_dtheta_function(z_, t_, ode_parameters, nn_parameters)
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
        # Combine the trajectories on the batches to one trajectory
        z = np.vstack(zs)
    else:
        z = hybrid_euler(z0, t, ode_parameters, nn_parameters) # 0.01-0.06s
        loss = J(z, z_ref, ode_parameters, nn_parameters)
        a0 = np.zeros(z0.shape)
        adjoint = adjoint_euler(a0, np.flip(z, axis=0), np.flip(z_ref, axis=0), np.flip(t), ode_parameters, nn_parameters)
        adjoint = np.flip(adjoint, axis=0)
        # Calculate the gradient of the hybrid ode with respect to the nn_parameters
        df_dtheta_trajectory = vectorized_df_dtheta_function(z, t, ode_parameters, nn_parameters)
        # Matrix multiplication of adjoint variable with jacobian
        df_dtheta_trajectory = unfreeze(df_dtheta_trajectory)
        for layer in df_dtheta_trajectory['params']:
            # Sum the matmul result over the entire time_span to get the final gradients
            df_dtheta_trajectory['params'][layer]['bias'] = np.einsum("iN,iNj->j", adjoint, df_dtheta_trajectory['params'][layer]['bias'])
            df_dtheta_trajectory['params'][layer]['kernel'] = np.einsum("iN,iNjk->jk", adjoint, df_dtheta_trajectory['params'][layer]['kernel'])
        df_dtheta = df_dtheta_trajectory
        dJ_dtheta = df_dtheta
        flat_dJ_dtheta, _ = flatten_util.ravel_pytree(dJ_dtheta)

    # Calculate the Loss on the trajectory over the validation data
    z_val = hybrid_euler(z_ref[-1], t_val, ode_parameters, nn_parameters)
    val_loss = J(z_val, z_ref_val, ode_parameters, nn_parameters)

    # If the training stagantes we could give the parameters a brownian noise kick
    if random_shift:
        if np.abs(loss - losses[-1]) < 0.1:
            flat_dJ_dtheta += np.random.normal(0, np.linalg.norm(flat_dJ_dtheta,2), flat_dJ_dtheta.shape)

    # Plot the results in regular intervals
    if epoch % checkpoint_interval == 0:
        # Due to batching the length of z can change
        plot_results(t[:z.shape[0]], z, z_ref[:z.shape[0]], os.path.join(results_directory,f'epoch_{epoch}'))
        plot_results(t_val, z_val, z_ref_val, os.path.join(results_directory,f'val_epoch_{epoch}'))

    # Save the best parameters measured on the validation data
    if epoch > 0 and val_loss < best_loss:
        optimizer_args['saved_nn_parameters'] = nn_parameters
        optimizer_args['best_loss'] = loss
        save_args = orbax_utils.save_args_from_target(nn_parameters)
        checkpoint_manager.save(epoch, nn_parameters, save_kwargs={'save_args': save_args})

    # Regularise the optimizer with a punishment on the nn_parameters (bigger parameter values
    # equals more a higher loss punishment)
    if lambda_ == 0.0:
        L2_regularisation = 0.0
    else:
        L2_regularisation = lambda_/(2*(t.shape[0])) * np.linalg.norm(flat_nn_parameters, 2)**2

    optimizer_args['epoch'] += 1
    optimizer_args['losses'].append(loss)
    optimizer_args['val_losses'].append(val_loss)

    end = time.time()
    print(f'Epoch: {epoch}, Loss: {loss:.5E}, Validation Loss {val_loss:.5E}, Time: {end-start:3.3f}')

    if val_loss < loss_cutoff:
        warnings.warn('Terminating Optimization: Required Loss reached')

    return loss + L2_regularisation, flat_dJ_dtheta/(t.shape[0])

def create_results_directory(directory, results_name, restore, overwrite):
    results_directory = os.path.join(directory, args.results_name)
    if os.path.exists(results_directory):
        if restore:
            print(f'Restoring parameters from previous run with Name: {args.results_name}')
        else:
            if overwrite:
                print(f'Deleting previous run with Name: {results_name}')
                shutil.rmtree(results_directory)
                os.mkdir(results_directory)
            else:
                print(f'Run with name {results_name} already exists and are not to be restored or overwritten')
                exit()
    else:
        os.mkdir(results_directory)
    return results_directory

def create_results_subdirectories(results_directory, a=False, r=False, c=True):
    if a:
        adjoint_directory = os.path.join(results_directory, 'adjoint')
        if not os.path.exists(adjoint_directory):
            os.mkdir(adjoint_directory)
    else:
        adjoint_directory = None

    if r:
        residual_directory = os.path.join(results_directory, 'residual')
        if not os.path.exists(residual_directory):
            os.mkdir(residual_directory)
    else:
        residual_directory = None

    if c:
        checkpoint_directory = os.path.join(results_directory, 'ckpt')
        if not os.path.exists(checkpoint_directory):
            os.mkdir(checkpoint_directory)
    else:
        checkpoint_directory = None

    return adjoint_directory, residual_directory, checkpoint_directory

def create_checkpoint_manager(checkpoint_directory, max_to_keep=1, create=True):
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=5, create=True)
    checkpoint_manager = orbax.checkpoint.CheckpointManager(checkpoint_directory, orbax_checkpointer, options)
    return checkpoint_manager

def post_processing(z0, t, z_ref, z0_val, t_val, z_ref_val, reference_ode_parameters, directory, optimizer_args):
    nn_parameters = optimizer_args['saved_nn_parameters']
    z = hybrid_euler(z0, t, reference_ode_parameters, nn_parameters)
    z_val = hybrid_euler(z0_val, t_val, reference_ode_parameters, nn_parameters)
    loss = J(z, z_ref, reference_ode_parameters, nn_parameters)
    val_loss = J(z_val, z_ref_val, reference_ode_parameters, nn_parameters)

    print(f'Best Loss in Training: {loss}, Validation Loss: {val_loss}')

    plot_results(t, z, z_ref, os.path.join(directory, 'best_training'))
    plot_results(t_val, z_val, z_ref_val, os.path.join(directory, 'best_validation'))

    # For plotting purposes we want to remove nan, inf, ... values from the loss data
    plot_losses(range(optimizer_args['epoch']),
                optimizer_args['losses'],
                optimizer_args['val_losses'],
                os.path.join(directory, 'losses'))

    if optimizer_args['residual_losses'] != []:
        plot_losses(range(optimizer_args['epoch']),
                    optimizer_args['residual_losses'],
                    path=os.path.join(directory, 'residual_losses'))

    return loss, val_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # ODE OPTIONS
    parser.add_argument('--kappa', type=float, default=1.0, help='oscillation constant of the VdP oscillation term')
    parser.add_argument('--mu', type=float, default=8.53, help='damping constant of the VdP damping term')
    parser.add_argument('--mass', type=float, default=1.0, help='mass constant of the VdP system')
    parser.add_argument('--start', type=float, default=0.0, help='Start value of the ODE integration')
    parser.add_argument('--end', type=float, default=100.0, help='End value of the ODE integration')
    parser.add_argument('--n_steps', type=float, default=5001, help='How many integration steps to perform')
    parser.add_argument('--aug_state', type=bool, default=False, help='Whether or not to use the augemented state for the ODE dynamics')
    parser.add_argument('--aug_dim', type=int, default=4, help='Number of augment dimensions')
    parser.add_argument('--stimulate', type=bool, default=True, help='Whether or not to use the stimulated dynamics')
    parser.add_argument('--simple_problem', type=bool, default=False, help='Whether or not to use a simple damped oscillator instead of VdP')

    # NEURAL NETWORK OPTIONS
    parser.add_argument('--layers', type=int, default=2, help='Number of hidden layers')
    parser.add_argument('--layer_size', type=int, default=10, help='Number of neurons in a hidden layer')

    # OPTIMIZER OPTIONS
    parser.add_argument('--method', type=str, default='BFGS', help='Which optimisation method to use')
    parser.add_argument('--tol', type=float, default=1e-10, help='Tolerance for the optimisation method')
    parser.add_argument('--opt_steps', type=float, default=1000, help='Max Number of steps for the Training')
    parser.add_argument('--random_shift', type=bool, default=False, help='Whether or not to shift the gradient of training stagnates')
    parser.add_argument('--batching', type=bool, default=True, help='whether or not to batch the training data')
    parser.add_argument('--n_batches', type=int, default=40, help='How many (arbitrary) batches to create')
    parser.add_argument('--batch_size', type=int, default=200, help='batch size (for samples-level batching)')
    parser.add_argument('--clean_batching', type=bool, default=True, help='Whether or not to split training data into with no overlap')
    parser.add_argument('--n_clean_batches', type=int, default=10, help='How many clean batches to create')

    # TRANSFER LEARNING (RESIDUAL TRAINING) OPTIONS
    parser.add_argument('--transfer_learning', type=bool, default=True, help='Tolerance for the optimisation method')
    parser.add_argument('--res_steps', type=float, default=1000, help='Number of steps for the Pretraining on the Residuals')

    # MISC OPTIONS
    parser.add_argument('--results_name', required=False, type=str, default='plot_test',
                        help='name under which the results should be saved, like plots and such, ignored for DoE')
    parser.add_argument('--lambda_', type=int, default=0, help='lambda in the L2 regularisation term')
    parser.add_argument('--loss_cutoff', type=float, default=1e-3, help='lower bound for validation loss after which training is stopped')
    parser.add_argument('--build_plot', required=False, default=True, action='store_true',
                        help='specify to build loss and accuracy plot')

    parser.add_argument('--checkpoint_interval', required=False, type=int, default=100,
                        help='path to save the resulting plot')
    parser.add_argument('--restore', required=False, type=bool, default=False,
                        help='restore previous parameters')
    parser.add_argument('--overwrite', required=False, type=bool, default=True,
                        help='overwrite previous result')

    # DESIGN OF EXPERIMENTS OPTIONS
    parser.add_argument('--doe_residual', required=False, type=bool, default=True,
                        help='Perform Design of Experiments with NN Parameters and ODE Hyperparameters on Residuals')
    parser.add_argument('--doe_trajectory', required=False, type=bool, default=True,
                        help='Perform Design of Experiments with NN Parameters and ODE Hyperparameters on Trajectory')
    parser.add_argument('--doe_title', required=False, type=str, default='NN_parameter_transfer',
                        help='Name for the DoE')

    parser.add_argument('--eval_freq', type=int, default=100, help='evaluate test accuracy every EVAL_FREQ '
                                                                   'samples-level batches')
    args = parser.parse_args()

    path = os.path.abspath(__file__)
    directory = os.path.sep.join(path.split(os.path.sep)[:-1])
    file_path = get_file_path(path)

    if not (args.doe_residual or args.doe_trajectory):

        results_directory = create_results_directory(directory=directory,
                                                     results_name=args.results_name,
                                                     restore=args.restore,
                                                     overwrite=args.overwrite)

        with open(os.path.join(results_directory, 'Arguments'), 'a') as file:
            file.write(str(args))

        adjoint_directory, residual_directory, checkpoint_directory = create_results_subdirectories(results_directory=results_directory, a=True, r=True)

        checkpoint_manager = create_checkpoint_manager(checkpoint_directory=checkpoint_directory,
                                                       max_to_keep=5)

        reference_ode_parameters = np.asarray([args.kappa, args.mu, args.mass])

        # Generate the reference data for training
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
        t = np.linspace(args.start, args.end, args.n_steps)
        z_ref = f_euler(z0, t, reference_ode_parameters)

        # Generate the reference data for validation
        z0_val = z_ref[-1]
        t_val = np.linspace(args.end, (args.end-args.start) * 1.5, int(args.n_steps * 0.5))
        z_ref_val = f_euler(z0_val, t_val, reference_ode_parameters)

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
        optimizer_args = {'time': t,
                          'val_time': t_val,
                          'initial_condition': z0,
                          'reference_solution': z_ref,
                          'validation_solution': z_ref_val,
                          'reference_ode_parameters': reference_ode_parameters,
                          'unravel_function': unravel_pytree,
                          'epoch': epoch,
                          'losses': [],
                          'val_losses': [],
                          'residual_losses': [],
                          'batching': args.batching,
                          'n_batches': args.n_batches,
                          'batch_size': args.batch_size,
                          'clean_batching': args.clean_batching,
                          'n_clean_batches': args.n_clean_batches,
                          'random_shift': args.random_shift,
                          'checkpoint_interval': args.checkpoint_interval,
                          'results_directory': residual_directory,
                          'saved_nn_parameters': nn_parameters,
                          'best_loss': np.inf,
                          'checkpoint_manager': checkpoint_manager,
                          'lambda': args.lambda_,
                          'loss_cutoff': args.loss_cutoff}


        # Train on Residuals
        ####################################################################################
        # Optimisers: CG, BFGS, Newton-CG, L-BFGS-B, TNC, SLSQP, dogleg, trust-ncg, trust-krylov, trust-exact and trust-constr
        if args.transfer_learning:
            try:
                residual_result = minimize(residual_wrapper, flat_nn_parameters, method=args.method, jac=True, args=optimizer_args, options={'maxiter':args.res_steps})
            except (KeyboardInterrupt, UserWarning):
                pass
            nn_parameters = optimizer_args['saved_nn_parameters']

            post_processing(z0, t, z_ref, z0_val, t_val, z_ref_val, reference_ode_parameters, residual_directory, optimizer_args)

            # # Plot the result of the training on residuals
            # z = hybrid_euler(z0, t, reference_ode_parameters, nn_parameters)
            # z_val = hybrid_euler(z0_val, t_val, reference_ode_parameters, nn_parameters)
            # plot_results(t, z, z_ref, residual_directory+'_best_training')
            # plot_results(t_val, z_val, z_ref_val, residual_directory+'_best_validation')
            # plot_losses(range(optimizer_args['epoch']),
            #             optimizer_args['losses'],
            #             optimizer_args['val_losses'],
            #             residual_directory+'_losses')


        # Train on Trajectory
        ####################################################################################
        flat_nn_parameters, _ = flatten_util.ravel_pytree(optimizer_args['saved_nn_parameters'])
        optimizer_args['epoch'] = 0
        optimizer_args['results_directory'] = adjoint_directory
        optimizer_args['losses'] = []
        optimizer_args['val_losses'] = []
        optimizer_args['residual_losses'] = []
        try:
            if args.opt_steps == 0:
                res = minimize(function_wrapper, flat_nn_parameters, method='BFGS', jac=True, args=optimizer_args, tol=args.tol)
            else:
                res = minimize(function_wrapper, flat_nn_parameters, method='BFGS', jac=True, args=optimizer_args, tol=args.tol, options={'maxiter':args.opt_steps})
            print(res)
        except (KeyboardInterrupt, UserWarning):
            pass

        # Print the result and generate trajectory and loss plots
        post_processing(z0, t, z_ref, z0_val, t_val, z_ref_val, reference_ode_parameters, adjoint_directory, optimizer_args)

    else:
        # Perform Design of Experiment
        if args.doe_title is None:
            now = datetime.datetime.now()
            doe_date = '-'.join([str(now.year), str(now.month), str(now.day)]) + '_' + '-'.join([str(now.hour), str(now.minute)])
            doe_directory = os.path.join(directory, doe_date)
        else:
            doe_directory = os.path.join(directory, args.doe_title)
            if not os.path.exists(doe_directory):
                os.mkdir(doe_directory)
            else:
                count = 1
                while os.path.exists(doe_directory):
                    doe_directory = os.path.join(directory, args.doe_title + f'_{count}')
                    count += 1
                os.mkdir(doe_directory)
        # else:
        #     print('DoE directory already exists, Date and Time is used as directory name instead')
        #     now = datetime.datetime.now()
        #     doe_date = '-'.join([str(now.year), str(now.month), str(now.day)]) + '_' + '-'.join([str(now.hour), str(now.minute)])
        #     doe_directory = os.path.join(directory, doe_date)
        #     os.mkdir(doe_directory)

        doe_results_file_path = os.path.join(doe_directory, 'doe_results.txt')
        doe_best_setup_residual_file = os.path.join(doe_directory, 'best_setup_residual.yaml')
        doe_best_setup_trajectory_file = os.path.join(doe_directory, 'best_setup_trajectory.yaml')

        with open(doe_results_file_path, 'a') as file:
            file.writelines(f'VdP Setup: kappa: {args.kappa}, mu: {args.mu}, mass: {args.mass}, Start: {args.start}, End: {args.end}, Steps: {args.n_steps}')
            file.write('\n')

        best_experiment_checkpoint_directory = os.path.join(doe_directory, 'best_result_ckpt')
        if not os.path.exists(best_experiment_checkpoint_directory):
            os.mkdir(best_experiment_checkpoint_directory)

        if args.doe_residual:

            # residual_doe_parameters = OrderedDict({'lambda': [0.0, 1.0, 2.0],
            #                                        'layers': [1, 2, 3],
            #                                        'l_size': [10, 20, 40]})

            residual_doe_parameters = OrderedDict({'lambda': [0.0],
                                                   'layers': [1],
                                                   'l_size': [10, 20]})

            levels = [len(val) for val in residual_doe_parameters.values()]

            doe = fullfact(levels)

            experiment_losses = []
            experiment_strings = []
            best_experiment = {'n_exp': None,
                               'setup': {},
                               'loss': np.inf,
                               'val_loss': np.inf,
                               'nn_parameters': None,
                               'time': 0.0}
            for n_exp, experiment in enumerate(doe):
                start = time.time()

                experiment_directory = os.path.join(doe_directory, f'Residual Experiment {n_exp}')
                if not os.path.exists(experiment_directory):
                    os.mkdir(experiment_directory)

                current_experiment_dict = {}
                for i, key in enumerate(residual_doe_parameters.keys()):
                    current_experiment_dict[key] = residual_doe_parameters[key][int(experiment[i])]

                experiment_strings.append(f'Residual Experiment {n_exp} - {current_experiment_dict}')

                print(experiment_strings[-1])

                adjoint_directory, residual_directory, checkpoint_directory = create_results_subdirectories(experiment_directory, r=True)

                checkpoint_manager = create_checkpoint_manager(checkpoint_directory=checkpoint_directory,
                                                               max_to_keep=1)

                reference_ode_parameters = np.asarray([args.kappa, args.mu, args.mass])

                # Generate the reference data for training
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

                t = np.linspace(args.start, args.end, args.n_steps)
                z_ref = f_euler(z0, t, reference_ode_parameters)

                z0_val = z_ref[-1]
                t_val = np.linspace(args.end, (args.end-args.start) * 1.5, int(args.n_steps * 0.5))
                z_ref_val = f_euler(z0_val, t_val, reference_ode_parameters)

                layers = [current_experiment_dict['l_size']]*current_experiment_dict['layers']
                layers.append(1)
                jitted_neural_network, nn_parameters = create_nn(layers, z0)

                flat_nn_parameters, unravel_pytree = flatten_util.ravel_pytree(nn_parameters)
                epoch = 0
                # Put all arguments the optimization needs into one array for the minimize function
                optimizer_args = {'time': t,
                                'val_time': t_val,
                                'initial_condition': z0,
                                'reference_solution': z_ref,
                                'validation_solution': z_ref_val,
                                'reference_ode_parameters': reference_ode_parameters,
                                'unravel_function': unravel_pytree,
                                'epoch': epoch,
                                'losses': [],
                                'val_losses': [],
                                'residual_losses': [],
                                'batching': args.batching,
                                'n_batches': args.n_batches,
                                'batch_size': args.batch_size,
                                'clean_batching': args.clean_batching,
                                'n_clean_batches': args.n_clean_batches,
                                'random_shift': args.random_shift,
                                'checkpoint_interval': args.checkpoint_interval,
                                'results_directory': residual_directory,
                                'saved_nn_parameters': nn_parameters,
                                'best_loss': np.inf,
                                'checkpoint_manager': checkpoint_manager,
                                'lambda': current_experiment_dict['lambda'],
                                'loss_cutoff': args.loss_cutoff}

                try:
                    residual_result = minimize(residual_wrapper, flat_nn_parameters, method=args.method, jac=True, args=optimizer_args, options={'maxiter':args.res_steps})
                except (KeyboardInterrupt, UserWarning):
                    pass
                nn_parameters = optimizer_args['saved_nn_parameters']
                loss, val_loss = post_processing(z0, t, z_ref, z0_val, t_val, z_ref_val, reference_ode_parameters, residual_directory, optimizer_args)

                experiment_time = time.time()-start

                experiment_strings[-1] = experiment_strings[-1] + f', Training loss: {loss:3.10f}, Validation loss: {val_loss:3.10f}, Time: {experiment_time:3.3f}'

                with open(doe_results_file_path, 'a') as file:
                    file.writelines(experiment_strings[-1])
                    file.write('\n')

                if val_loss < best_experiment['val_loss']:
                    best_experiment['n_exp'] = n_exp
                    best_experiment['setup'] = current_experiment_dict
                    best_experiment['loss'] = loss
                    best_experiment['val_loss'] = val_loss
                    best_experiment['nn_parameters'] = nn_parameters
                    best_experiment['time'] = experiment_time
                    checkpoint_manager = create_checkpoint_manager(checkpoint_directory=best_experiment_checkpoint_directory,
                                                                   max_to_keep=1)
                    save_args = orbax_utils.save_args_from_target(nn_parameters)
                    checkpoint_manager.save(n_exp, nn_parameters, save_kwargs={'save_args': save_args})

            # best_n = best_experiment['n_exp']
            # best_setup = best_experiment['setup']
            # best_loss = best_experiment['loss']
            # best_val_loss = best_experiment['val_loss']
            # best_time = best_experiment['time']
            # best_experiment_string = f'Experiment {best_n} - {best_setup}, Training loss: {best_loss:3.10f}, Validation loss: {best_val_loss:3.10f}, Time: {best_time:3.3f}'
            # # Now we want to determine the best Experiment and store the associated parameters in a special place
            # # so we can quickly access them
            # with open(doe_results_file_path, 'a') as file:
            #     file.writelines(best_experiment_string)
            #     file.write('\n')

            # Clean up the best experiment for yaml dumping
            yaml_dict = best_experiment.copy()
            yaml_dict['loss'] = float(best_experiment['loss'])
            yaml_dict['val_loss'] = float(best_experiment['val_loss'])
            del yaml_dict['nn_parameters']

            with open(doe_best_setup_residual_file, 'w') as file:
                yaml.dump(yaml_dict, file)

        if args.doe_trajectory:
            if args.doe_residual:
                # Restore parameters of residual run
                checkpoint_manager = create_checkpoint_manager(checkpoint_directory=best_experiment_checkpoint_directory,
                                                               max_to_keep=1)
                step = checkpoint_manager.latest_step()
                nn_parameters = checkpoint_manager.restore(step)

                # Restore the rest of the results of the residual run
                with open(doe_best_setup_residual_file, 'r') as file:
                    best_experiment = yaml.safe_load(file)

                layers = best_experiment['setup']['layers']
                layer_size = best_experiment['setup']['l_size']
                lambda_ = best_experiment['setup']['lambda']
                trajectory_doe_parameters = OrderedDict({
                    'n_clean_batches': [10, 20, 40],
                    'lambda': [lambda_],
                    'layers': [layers],
                    'l_size': [layer_size],
                })
                best_experiment['nn_parameters'] = nn_parameters
            else:
                best_experiment = {'n_exp': None, 'setup': {}, 'loss': np.inf, 'val_loss': np.inf, 'nn_parameters': None, 'time': 0.0}
                trajectory_doe_parameters = OrderedDict({
                    'n_clean_batches': [10, 20, 40],
                    'lambda': [0.0, 1.0, 2.0],
                    'layers': [1, 2, 3],
                    'l_size': [10, 20, 30]
                })

            levels = [len(val) for val in trajectory_doe_parameters.values()]

            doe = fullfact(levels)

            experiment_losses = []
            experiment_strings = []
            for n_exp, experiment in enumerate(doe):
                start = time.time()

                experiment_directory = os.path.join(doe_directory, f' Trajectory Experiment {n_exp}')
                if not os.path.exists(experiment_directory):
                    os.mkdir(experiment_directory)

                current_experiment_dict = {}
                for i, key in enumerate(trajectory_doe_parameters.keys()):
                    current_experiment_dict[key] = trajectory_doe_parameters[key][int(experiment[i])]

                experiment_strings.append(f'Trajectory Experiment {n_exp} - {current_experiment_dict}')

                print(experiment_strings[-1])

                adjoint_directory, residual_directory, checkpoint_directory = create_results_subdirectories(experiment_directory, a=True)

                checkpoint_manager = create_checkpoint_manager(checkpoint_directory=checkpoint_directory,
                                                               max_to_keep=1)

                reference_ode_parameters = np.asarray([args.kappa, args.mu, args.mass])

                z0 = np.array([1.0, 0.0])
                t = np.linspace(args.start, args.end, args.n_steps)
                z_ref = f_euler(z0, t, reference_ode_parameters)

                z0_val = z_ref[-1]
                t_val = np.linspace(args.end, (args.end-args.start) * 1.5, int(args.n_steps * 0.5))
                z_ref_val = f_euler(z0_val, t_val, reference_ode_parameters)

                layers = [current_experiment_dict['l_size']]*current_experiment_dict['layers']
                layers.append(1)
                if args.doe_residual:
                    jitted_neural_network, _ = create_nn(layers, z0)
                else:
                    jitted_neural_network, nn_parameters = create_nn(layers, z0)

                flat_nn_parameters, unravel_pytree = flatten_util.ravel_pytree(nn_parameters)
                # Put all arguments the optimization needs into one array for the minimize function
                residual_outputs = create_residual_references(z_ref, t, reference_ode_parameters)[:,1]
                optimizer_args = {'time': t,
                                'val_time': t_val,
                                'initial_condition': z0,
                                'reference_solution': z_ref,
                                'validation_solution': z_ref_val,
                                'reference_ode_parameters': reference_ode_parameters,
                                'unravel_function': unravel_pytree,
                                'epoch': 0,
                                'losses': [],
                                'val_losses': [],
                                'batching': args.batching,
                                'n_batches': args.n_batches,
                                'batch_size': args.batch_size,
                                'clean_batching': args.clean_batching,
                                'n_clean_batches': current_experiment_dict['n_clean_batches'],
                                'random_shift': args.random_shift,
                                'checkpoint_interval': args.checkpoint_interval,
                                'results_directory': adjoint_directory,
                                'saved_nn_parameters': nn_parameters,
                                'best_loss': np.inf,
                                'checkpoint_manager': checkpoint_manager,
                                'lambda': current_experiment_dict['lambda'],
                                'loss_cutoff': args.loss_cutoff,
                                'residual_outputs': residual_outputs}

                # Train on Trajectory
                ####################################################################################
                try:
                    res = minimize(function_wrapper, flat_nn_parameters, method='BFGS', jac=True, args=optimizer_args, tol=args.tol, options={'maxiter': args.opt_steps})
                    print(res)
                except (KeyboardInterrupt, UserWarning):
                    pass
                nn_parameters = optimizer_args['saved_nn_parameters']
                loss, val_loss = post_processing(z0, t, z_ref, z0_val, t_val, z_ref_val, reference_ode_parameters, adjoint_directory, optimizer_args)

                experiment_time = time.time()-start

                experiment_strings[-1] = experiment_strings[-1] + f', Training loss: {loss:3.10f}, Validation loss: {val_loss:3.10f}, Time: {experiment_time:3.3f}'

                with open(doe_results_file_path, 'a') as file:
                    file.writelines(experiment_strings[-1])
                    file.write('\n')

                if val_loss < best_experiment['val_loss']:
                    best_experiment['n_exp'] = n_exp
                    best_experiment['setup'] = current_experiment_dict
                    best_experiment['loss'] = loss
                    best_experiment['val_loss'] = val_loss
                    best_experiment['nn_parameters'] = nn_parameters
                    best_experiment['time'] = experiment_time
                    checkpoint_manager = create_checkpoint_manager(checkpoint_directory=best_experiment_checkpoint_directory,
                                                                   max_to_keep=1)
                    save_args = orbax_utils.save_args_from_target(nn_parameters)
                    checkpoint_manager.save(0, nn_parameters, save_kwargs={'save_args': save_args})

                yaml_dict = best_experiment.copy()
                yaml_dict['loss'] = float(best_experiment['loss'])
                yaml_dict['val_loss'] = float(best_experiment['val_loss'])
                del yaml_dict['nn_parameters']

                with open(doe_best_setup_trajectory_file, 'w') as file:
                    yaml.dump(yaml_dict, file)

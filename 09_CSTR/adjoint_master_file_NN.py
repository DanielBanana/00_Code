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
import warnings
import logging

# To use the plot_results file we need to add the uppermost folder to the PYTHONPATH
# Only Works if file gets called from 00_Code
sys.path.insert(0, os.getcwd())
from plot_results import plot_results, plot_losses, get_file_path
from utils import build_plot, result_plot_multi_dim, create_results_directory, create_results_subdirectories, create_doe_experiments, create_experiment_directory
import yaml

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

class AdamOptim():
    def __init__(self, eta=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.m_dp, self.v_dp = 0, 0
        self.m_db, self.v_db = 0, 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.eta = eta
    def update(self, t, p, dp):
        ## dw, db are from current minibatch
        ## momentum beta 1
        # *** weights *** #
        self.m_dp = self.beta1*self.m_dp + (1-self.beta1)*dp

        ## rms beta 2
        # *** weights *** #
        self.v_dp = self.beta2*self.v_dp + (1-self.beta2)*(dp**2)

        ## bias correction
        m_dp_corr = self.m_dp/(1-self.beta1**t)
        v_dp_corr = self.v_dp/(1-self.beta2**t)

        ## update weights and biases
        p = p - self.eta*(m_dp_corr/(np.sqrt(v_dp_corr)+self.epsilon))
        return p

@jit
def ode(z, t, ode_parameters):
    '''Calculates the right hand side of the original ODE.'''

    q = ode_parameters[0]  # L/min
    cA_i = ode_parameters[1]  # mol/L
    T_i = ode_parameters[2]  # K
    V = ode_parameters[3]  # L
    rho = ode_parameters[4] # g/L
    C = ode_parameters[5] # J/(g K)
    Hr = ode_parameters[6]  # J/(g K)
    E_over_R = ode_parameters[7]  # K
    k0 = ode_parameters[8]  # 1/min
    UA = ode_parameters[9]  # J/(min K)
    Tc = ode_parameters[10]

    k = k0*jnp.exp(-E_over_R/z[1])
    w = q*rho
    derivative = jnp.array([q*(cA_i - z[0])/V - k*z[0],
                            1/(V*rho*C)*(w*C*(T_i - z[1]) - Hr*V*k*z[0] + UA*(Tc - z[1]))])
    return derivative

@jit
def ode_res(z, t, ode_parameters):
    '''Calculates the right hand side of the deficient ODE.'''
    q = ode_parameters[0]  # L/min
    cA_i = ode_parameters[1]  # mol/L
    T_i = ode_parameters[2]  # K
    V = ode_parameters[3]  # L
    rho = ode_parameters[4] # g/L
    C = ode_parameters[5] # J/(g K)
    Hr = ode_parameters[6]  # J/(g K)
    E_over_R = ode_parameters[7]  # K
    k0 = ode_parameters[8]  # 1/min
    UA = ode_parameters[9]  # J/(min K)
    Tc = ode_parameters[10]


    w = q*rho
    derivative = jnp.array([q*(cA_i - z[0])/V,
                            1/(V*rho*C)*(w*C*(T_i - z[1]) + UA*(Tc - z[1]))])
    return derivative

@jit
def hybrid_ode(z, t, ode_parameters, nn_parameters):
    '''Calculates the right hand side of the hybrid ODE, where
    the damping term is replaced by the neural network'''
    q = ode_parameters[0]  # L/min
    cA_i = ode_parameters[1]  # mol/L
    T_i = ode_parameters[2]  # K
    V = ode_parameters[3]  # L
    rho = ode_parameters[4] # g/L
    C = ode_parameters[5] # J/(g K)
    Hr = ode_parameters[6]  # J/(g K)
    E_over_R = ode_parameters[7]  # K
    k0 = ode_parameters[8]  # 1/min
    UA = ode_parameters[9]  # J/(min K)
    Tc = ode_parameters[10]

    k = jitted_neural_network(nn_parameters, z)
    w = q*rho
    derivative = jnp.array([q*(cA_i - z[0])/V - jitted_neural_network(nn_parameters, z),
                            1/(V*rho*C)*(w*C*(T_i - z[1]) - Hr*V*jitted_neural_network(nn_parameters, z) + UA*(Tc - z[1]))]).flatten()
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

def adjoint_euler(a0, z, z_ref, t, ode_parameters, nn_parameters):
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
def create_mini_batch(n_batches, batch_size, X, adjoint, t):

    n_timesteps = t.shape[0]
    # Create batches of trajectories
    if n_timesteps-batch_size == 0:
        s = np.array([0])
    else:
        s = np.random.choice(range(n_timesteps-batch_size), n_batches)

    x0 = X[s, :]
    a0 = adjoint[s, :]
    targets = np.empty((n_batches, batch_size, X.shape[1]))
    adjoints = np.empty((n_batches, batch_size, adjoint.shape[1]))
    ts = []

    for i in range(n_batches):
        targets[i, 0:batch_size, :] = X[s[i] + 0:(s[i] + batch_size), :]
        adjoints[i, 0:batch_size, :] = adjoint[s[i] + 0:(s[i] + batch_size), :]
        ts.append(t[s[i]:s[i]+batch_size])

    return x0, targets, a0, adjoints, ts

def create_clean_mini_batch(n_batches, X, t):
    n_timesteps = t.shape[0]
    # Create batches of trajectories
    mini_batch_size = int(n_timesteps/n_batches)
    s = [mini_batch_size * i for i in range(n_batches)]
    x0 = X[s, :]
    # a0 = adjoint[s, :]
    targets = np.empty((n_batches, mini_batch_size, X.shape[1]))
    # adjoints = np.empty((n_batches, mini_batch_size, adjoint.shape[1]))
    ts = []
    for i in range(n_batches):
        targets[i, 0:mini_batch_size, :] = X[s[i] + 0:(s[i] + mini_batch_size), :]
        # adjoints[i, 0:mini_batch_size, :] = adjoint[s[i] + 0:(s[i] + mini_batch_size), :]
        ts.append(t[s[i]:s[i]+mini_batch_size])
    return x0, targets, ts

def model_losses(z0, a0s, ts, ode_parameters, nn_parameters, targets):
    zs = []
    losses = []
    adjoints = []
    # Compute Predictions
    for i, ic in enumerate(z0):
        z = hybrid_euler(ic, ts[i], ode_parameters, nn_parameters)
        adjoint = adjoint_euler(a0s[i], np.flip(z, axis=0), np.flip(targets[i], axis=0), np.flip(ts[i]), ode_parameters, nn_parameters)
        adjoint = np.flip(adjoint, axis=0)
        losses.append(float(J(z, targets[i], ode_parameters, nn_parameters)))
        zs.append(z)
        adjoints.append(adjoint)

    return zs, adjoints, losses

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
        self.layers = [nn.Dense(feat, kernel_init=nn.initializers.normal(1.0), bias_init=nn.initializers.normal(1.0)) for feat in self.features]

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
                x = nn.relu(x)
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
    t_val = optimizer_args['val_time']
    z0 = optimizer_args['initial_condition']
    z_ref = optimizer_args['reference_solution']
    z_ref_val = optimizer_args['validation_solution']
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
    lambda_ = optimizer_args['lambda']
    logger = optimizer_args['logger']


    # Get the parameters of the neural network out of the array structure into the
    # tree structure
    nn_parameters = unravel_pytree(flat_nn_parameters)
    # Save best parameters

    # in this case we only want the second part since the first part is completly known
    # and the neural network only works on the second part
    outputs = create_residuals(z_ref, t, ode_parameters)[:,1]
    inputs = z_ref[:-1]

    z = hybrid_euler(z0, t, ode_parameters, nn_parameters)
    z_val = hybrid_euler(z_ref[-1], t_val, ode_parameters, nn_parameters)

    # loss = J_residual(inputs, outputs, nn_parameters)
    # res_loss = J_residual(inputs, outputs, nn_parameters)
    res_loss, gradient = jax.value_and_grad(J_residual, argnums=2)(inputs, outputs, nn_parameters)
    true_loss = float(J(z, z_ref, ode_parameters, nn_parameters))
    true_val_loss = float(J(z_val, z_ref_val, ode_parameters, nn_parameters))
    flat_gradient, _ = flatten_util.ravel_pytree(gradient)
    optimizer_args['epoch'] += 1
    optimizer_args['losses'].append(true_loss)
    optimizer_args['losses_res'].append(float(res_loss))
    optimizer_args['losses_val'].append(true_val_loss)
    if epoch > 0 and true_val_loss < best_loss:
        optimizer_args['saved_nn_parameters'] = nn_parameters
        optimizer_args['best_loss'] = true_loss
        optimizer_args['best_val_loss'] = true_val_loss
        save_args = orbax_utils.save_args_from_target(nn_parameters)
        checkpoint_manager.save(epoch, nn_parameters, save_kwargs={'save_args': save_args})

    if epoch % checkpoint_interval == 0:
        plot_results(t, z, z_ref, results_path+f'_epoch_{epoch}')
        plot_results(t_val, z_val, z_ref_val, results_path+f'_epoch_{epoch}_val')
        plot_losses(epochs=list(range(len(optimizer_args['losses']))), training_losses=optimizer_args['losses'], validation_losses=optimizer_args['losses_val'], path=residual_path+'_losses')
        plot_losses(epochs=list(range(len(optimizer_args['losses_res']))), training_losses=optimizer_args['losses_res'], path=residual_path+'_losses_res')

        # optimizer_args['saved_nn_parameters'] = nn_parameters

    L2_regularisation = lambda_/(2*(t.shape[0])) * np.linalg.norm(flat_nn_parameters, 2)**2
    # L2_regularisation = 0.0

    end = time.time()

    loss_cutoff = 1e-5
    if res_loss < loss_cutoff:
        warnings.warn('Terminating Optimization: Required Loss reached')

    logger.info(f'Pretraining: Epoch: {epoch}, Residual Loss: {res_loss:.10f}, True Loss: {true_loss:.10f}, Time: {end-start:3.3f}')
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
    clean_n_batches = optimizer_args['clean_n_batches']
    random_shift = optimizer_args['random_shift']
    checkpoint_interval = optimizer_args['checkpoint_interval']
    results_path = optimizer_args['results_path']
    best_loss = optimizer_args['best_loss']
    checkpoint_manager = optimizer_args['checkpoint_manager']
    lambda_ = optimizer_args['lambda']
    logger = optimizer_args['logger']

    # Get the parameters of the neural network out of the array structure into the
    # tree structure
    nn_parameters = unravel_pytree(flat_nn_parameters)

    z = hybrid_euler(z0, t, ode_parameters, nn_parameters) # 0.01-0.06s
    z = np.nan_to_num(z, nan=1e8)
    a0 = np.zeros(z0.shape)
    # df_dtheta_trajectory = vectorized_df_dtheta_function(z, t, ode_parameters, nn_parameters)
    adjoint = adjoint_euler(a0, np.flip(z, axis=0), np.flip(z_ref, axis=0), np.flip(t), ode_parameters, nn_parameters)
    adjoint = np.flip(adjoint, axis=0)

    if batching:
        # if clean_batching:
        #     # z0s, targets, a0s, adjoints_, ts = create_clean_mini_batch(clean_n_batches, z_ref, adjoint, t)
        # else:
        #     z0s, targets, a0s, adjoints_, ts = create_mini_batch(n_batches, batch_size, z_ref, adjoint, t)
        a0s = np.zeros((n_batches, *z0.shape))
        zs, adjoints, losses = model_losses(z0s, a0s, ts, ode_parameters, nn_parameters, targets)
        logger.info(f'Batch losses: {losses}')
        loss = np.asarray(losses).mean()
        logger.info(f'Mean Loss on Batches: {loss}')
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

    loss = float(J(z, z_ref, ode_parameters, nn_parameters))

    if random_shift:
        if np.abs(loss - losses[-1]) < 0.1:
            flat_dJ_dtheta += np.random.normal(0, np.linalg.norm(flat_dJ_dtheta,2), flat_dJ_dtheta.shape)

    z_val = hybrid_euler(z_ref[-1], t_val, ode_parameters, nn_parameters)
    val_loss = float(J(z_val, z_ref_val, ode_parameters, nn_parameters))



        # optimizer_args['saved_nn_parameters'] = nn_parameters

        # checkpoint_manager.save(epoch, nn_parameters, save_kwargs={'save_args': save_args})

    optimizer_args['epoch'] += 1
    optimizer_args['losses'].append(loss)
    optimizer_args['losses_val'].append(val_loss)

    if epoch > 0 and val_loss < best_loss:
        optimizer_args['saved_nn_parameters'] = nn_parameters
        optimizer_args['best_loss'] = loss
        optimizer_args['best_loss_val'] = val_loss
        save_args = orbax_utils.save_args_from_target(nn_parameters)
        checkpoint_manager.save(epoch, nn_parameters, save_kwargs={'save_args': save_args})

    end = time.time()

    logger.info(f'Epoch: {epoch}, Loss: {loss:.7f}, Time: {end-start:3.3f}')
    if epoch % checkpoint_interval == 0:
        plot_results(t, z, z_ref, results_path+f'_epoch_{epoch}')
        logger.info('#################################################')
        logger.info(f'Epoch: {epoch}, Valdiation Loss: {val_loss:.7f}')
        plot_results(t_val, z_val, z_ref_val, results_path+f'_epoch_{epoch}_val')
        plot_losses(epochs=list(range(len(optimizer_args['losses']))), training_losses=optimizer_args['losses'], validation_losses=optimizer_args['losses_val'], path=residual_path+'_losses')
    return loss, flat_dJ_dtheta/(t.shape[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--start', type=float, default=0.0, help='Start value of the ODE integration')
    parser.add_argument('--end', type=float, default=20.0, help='End value of the ODE integration')
    parser.add_argument('--n_steps', type=int, default=1001, help='How many integration steps to perform')

    parser.add_argument('--layers', type=int, default=1, help='Number of hidden layers')
    parser.add_argument('--layer_size', type=int, default=25, help='Number of neurons in a hidden layer')
    parser.add_argument('--lambda_', type=float, default=0.0, help='lambda in the L2 regularisation term')

    parser.add_argument('--aug_state', type=bool, default=False, help='Whether or not to use the augemented state for the ODE dynamics')
    parser.add_argument('--aug_dim', type=int, default=4, help='Number of augment dimensions')
    parser.add_argument('--random_shift', type=bool, default=False, help='Whether or not to shift the gradient of training stagnates')
    parser.add_argument('--batching', type=bool, default=False, help='whether or not to batch the training data')
    parser.add_argument('--n_batches', type=int, default=200, help='How many (arbitrary) batches to create')
    parser.add_argument('--batch_size', type=int, default=50, help='batch size (for samples-level batching)')
    parser.add_argument('--clean_batching', type=bool, default=False, help='Whether or not to split training data into with no overlap')
    parser.add_argument('--clean_n_batches', type=int, default=20, help='How many clean batches to create')

    parser.add_argument('--stimulate', type=bool, default=False, help='Whether or not to use the stimulated dynamics')
    parser.add_argument('--simple_problem', type=bool, default=False, help='Whether or not to use a simple damped oscillator instead of VdP')

    parser.add_argument('--method', type=str, default='Adam', help='Which optimisation method to use')
    parser.add_argument('--adam_eta', type=float, default=0.1, help='damping value of the VdP damping term')
    parser.add_argument('--adam_beta1', type=float, default=0.99, help='damping value of the VdP damping term')
    parser.add_argument('--adam_beta2', type=float, default=0.999, help='damping value of the VdP damping term')
    parser.add_argument('--adam_eps', type=float, default=1e-8, help='damping value of the VdP damping term')

    parser.add_argument('--tol', type=float, default=1e-6, help='Tolerance for the optimisation method')
    parser.add_argument('--opt_steps', type=int, default=1000, help='Max Number of steps for the Training')

    parser.add_argument('--transfer_learning', type=bool, default=True, help='Whether or not to use residual transfer learning')
    parser.add_argument('--res_steps', type=int, default=1000, help='Number of steps for the Pretraining on the Residuals')

    parser.add_argument('--build_plot', required=False, default=True, action='store_true',
                        help='specify to build loss and accuracy plot')
    parser.add_argument('--results_name', required=False, type=str, default='results',
                        help='name under which the results should be saved, like plots and such')
    parser.add_argument('--checkpoint_interval', required=False, type=int, default=50,
                        help='path to save the resulting plot')
    parser.add_argument('--restore', required=False, type=bool, default=False,
                        help='restore previous parameters')

    # FILE SETUP
    parser.add_argument('--results_file', type=str, default='doe_results.yaml')
    parser.add_argument('--results_directory', type=str, default=None)

    parser.add_argument('--eval_freq', type=int, default=25, help='evaluate test accuracy every EVAL_FREQ '
                                                                   'samples-level batches')
    args = parser.parse_args()

    path = os.path.abspath(__file__)
    directory = os.path.sep.join(path.split(os.path.sep)[:-1])
    if args.results_directory is None:
        args.results_directory = create_results_directory(directory=directory, results_directory_name=args.results_name)
        file_path = get_file_path(path)

    log_file = os.path.join(args.results_directory, 'GRAD.log')
    logging.basicConfig(filename=log_file, encoding='utf-8', level=logging.INFO)
    logger = logging.getLogger('GRAD')

    trajectory_directory = os.path.join(args.results_directory, 'trajectory')
    if not os.path.exists(trajectory_directory):
        os.mkdir(trajectory_directory)
    trajectory_path = os.path.join(trajectory_directory, args.results_name)

    residual_directory = os.path.join(args.results_directory, 'residual')
    if not os.path.exists(residual_directory):
        os.mkdir(residual_directory)
    residual_path = os.path.join(residual_directory, args.results_name)

    checkpoint_directory = os.path.join(args.results_directory, 'ckpt')
    if not os.path.exists(checkpoint_directory):
        os.mkdir(checkpoint_directory)

    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=5, create=True)
    checkpoint_manager = orbax.checkpoint.CheckpointManager(checkpoint_directory, orbax_checkpointer, options)

    # if args.aug_state:
    #     ode = ode_aug
    #     hybrid_ode = hybrid_ode_aug
    #     z0 = [1.0, 0.0]
    #     z0 += [0.0 for i in range(args.aug_dim)]
    #     z0 = np.asarray(z0)
    # elif args.stimulate:
    #     ode = ode_stim
    #     hybrid_ode = hybrid_ode_stim
    #     ode_res = ode_stim_res
    #     z0 = np.array([1.0, 0.0])
    # elif args.simple_problem:
    #     ode = ode_simple
    #     hybrid_ode = hybrid_ode_simple
    #     z0 = np.array([2.0, 0.0])
    # else:
    z0 = np.array([1.0, 300])

    t_ref = np.linspace(args.start, args.end, args.n_steps)
    t_val = np.linspace(args.end, (args.end-args.start) * 1.5, int(args.n_steps * 0.5))
    q = 100  # L/min
    cA_i = 1  # mol/L
    T_i = 350  # K
    V = 100  # L
    rho = 1000 # g/L
    C = 0.239 # J/(g K)
    Hr = -5e4  # J/(g K)
    E_over_R = 8750  # K
    k0 = 7.2e10  # 1/min
    UA = 5e4  # J/(min K)
    Tc = 300  # K
    reference_ode_parameters = np.asarray([q, cA_i, T_i, V, rho, C, Hr, E_over_R, k0, UA, Tc])

    z_ref = f_euler(z0, t_ref, reference_ode_parameters)
    z0_val = z_ref[-1]
    z_val = f_euler(z0_val, t_val, reference_ode_parameters)

    plot_results(t_ref, None, z_ref, trajectory_path+'_ref')

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
                    'val_time': t_val,
                    'initial_condition' : z0,
                    'reference_solution' : z_ref,
                    'validation_solution' : z_val,
                    'reference_ode_parameters' : reference_ode_parameters,
                    'unravel_function' : unravel_pytree,
                    'epoch' : epoch,
                    'losses' : [],
                    'losses_res': [],
                    'losses_val': [],
                    'losses_val_res': [],
                    'batching' : args.batching,
                    'n_batches': args.n_batches,
                    'batch_size': args.batch_size,
                    'clean_batching': args.clean_batching,
                    'clean_n_batches': args.clean_n_batches,
                    'random_shift' : args.random_shift,
                    'checkpoint_interval' : args.checkpoint_interval,
                    'results_path' : residual_path,
                    'saved_nn_parameters' : nn_parameters,
                    'best_loss' : np.inf,
                    'best_loss_val': np.inf,
                    'checkpoint_manager' : checkpoint_manager,
                    'lambda' : args.lambda_,
                    'logger': logger}

    if args.clean_batching:
        z0s, targets, ts = create_clean_mini_batch(args.clean_n_batches, z_ref, t_ref)
        optimizer_args['z0s'] = z0s
        optimizer_args['targets'] = targets
        optimizer_args['ts'] = ts
    else:
        optimizer_args['z0s'] = None
        optimizer_args['targets'] = None
        optimizer_args['ts'] = None

    # Train on Residuals
    ####################################################################################
    # Optimisers: CG, BFGS, Newton-CG, L-BFGS-B, TNC, SLSQP, dogleg, trust-ncg, trust-krylov, trust-exact and trust-constr

    results_dict = {}

    if args.transfer_learning:
        start = time.time()
        try:

            if args.method == 'Adam':
                Adam = AdamOptim(eta=args.adam_eta, beta1=args.adam_beta1, beta2=args.adam_beta2, epsilon=args.adam_eps)
                for i in range(args.res_steps):
                    loss, grad = residual_wrapper(flat_nn_parameters=flat_nn_parameters, optimizer_args=optimizer_args)
                    flat_nn_parameters = Adam.update(i+1, flat_nn_parameters, grad)
            else:
                residual_result = minimize(residual_wrapper, flat_nn_parameters, method=args.method, jac=True, args=optimizer_args, options={'maxiter':args.res_steps})
        except (KeyboardInterrupt, UserWarning):
            pass
        experiment_time = time.time() - start
        nn_parameters = optimizer_args['saved_nn_parameters']
        best_loss = optimizer_args['best_loss']
        logger.info(f'Best Loss in Residual Training: {best_loss}')
        flat_nn_parameters, _ = flatten_util.ravel_pytree(nn_parameters)
        # Plot the result of the training on residuals
        z = hybrid_euler(z0, t_ref, reference_ode_parameters, nn_parameters)
        plot_results(t_ref, z, z_ref, residual_path+'_best')
        plot_losses(epochs=list(range(len(optimizer_args['losses']))), training_losses=optimizer_args['losses'], validation_losses=optimizer_args['losses_val'], path=residual_path+'_losses')
        plot_losses(epochs=list(range(len(optimizer_args['losses_res']))), training_losses=optimizer_args['losses_res'], path=residual_path+'_losses_res')

        results_dict['losses_train_res'] =  list(optimizer_args['losses_res'])
        results_dict['time_res'] = experiment_time

    # Train on Trajectory
    ####################################################################################
    optimizer_args['results_path'] = trajectory_path
    start = time.time()
    try:
        if args.method == 'Adam':
            Adam = AdamOptim(eta=args.adam_eta*10, beta1=args.adam_beta1, beta2=args.adam_beta2, epsilon=args.adam_eps)
            for i in range(args.opt_steps):
                loss, grad = function_wrapper(flat_nn_parameters=flat_nn_parameters, optimizer_args=optimizer_args)
                flat_nn_parameters = Adam.update(i+1, flat_nn_parameters, grad)
        else:
            res = minimize(function_wrapper, flat_nn_parameters, method=args.method, jac=True, args=optimizer_args, tol=args.tol)
    except (KeyboardInterrupt, UserWarning):
        pass

    experiment_time = time.time() - start

    results_dict['losses_train_traj'] =  list(optimizer_args['losses'])
    results_dict['losses_train_traj_val'] = list(optimizer_args['losses_val'])
    results_dict['time_traj'] = experiment_time
    logger.info(f'Dumping results to {args.results_file}.')
    with open(args.results_file, 'w') as file:
        yaml.dump(results_dict, file)

    nn_parameters = optimizer_args['saved_nn_parameters']
    flat_nn_parameters, _ = flatten_util.ravel_pytree(nn_parameters)
    dumpable_params = []
    for param in flat_nn_parameters:
        dumpable_params.append(float(param))
    with open(args.results_file, 'a') as file:
        yaml.dump({'flat_parameters': dumpable_params}, file)

    best_loss = optimizer_args['best_loss']
    logger.info(f'Best Loss in Training: {best_loss}')

    z_training = hybrid_euler(z0, t_ref, reference_ode_parameters, nn_parameters)
    z_validation = hybrid_euler(z0_val, t_val, reference_ode_parameters, nn_parameters)

    plot_results(t_ref, z_training, z_ref, trajectory_path+'_best')
    plot_results(t_val, z_validation, z_val, trajectory_path+'_best_val')
    plot_losses(epochs=range(len(optimizer_args['losses'])), training_losses=optimizer_args['losses'], validation_losses=optimizer_args['losses_val'], path=trajectory_path)


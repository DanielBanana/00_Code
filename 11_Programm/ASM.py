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
from utils import build_plot, result_plot_multi_dim, create_results_directory, create_results_subdirectories, create_doe_experiments, create_experiment_directory, visualise_wb
import yaml

'''
Naming Conventions:
    x       refers to the state
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

# @jit
# def ode(x, t, variables):
#     '''Calculates the right hand side of the original ODE.'''
#     kappa = variables[0]
#     mu = variables[1]
#     mass = variables[2]
#     derivative = jnp.array([x[1],
#                            -kappa*x[0]/mass + (mu*(1-x[0]**2)*x[1])/mass])
#     return derivative

# @jit
# def ode_res(x, t, variables):
#     '''Calculates the right hand side of the deficient ODE.'''
#     kappa = variables[0]
#     mu = variables[1]
#     mass = variables[2]
#     derivative = jnp.array([x[1],
#                            -kappa*x[0]/mass])
#     return derivative

# @jit
# def hybrid_ode(x, t, variables, parameters):
#     '''Calculates the right hand side of the hybrid ODE, where
#     the damping term is replaced by the neural network'''
#     kappa = variables[0]
#     mu = variables[1]
#     mass = variables[2]
#     derivative = jnp.array([jnp.array((x[1],)),
#                             jnp.array((-kappa*x[0]/mass,)) + jitted_neural_network(parameters, x)]).flatten()
#     return derivative

# @jit
# def ode_stim(x, t, variables):
#     '''Calculates the right hand side of the original ODE.'''
#     kappa = variables[0]
#     mu = variables[1]
#     mass = variables[2]
#     derivative = jnp.array([x[1],
#                            -kappa*x[0]/mass + (mu*(1-x[0]**2)*x[1])/mass + 1.2*jnp.cos(0.628*t)])
#     return derivative

# @jit
# def ode_stim_res(x, t, variables):
#     '''Calculates the right hand side of the original ODE.'''
#     kappa = variables[0]
#     mu = variables[1]
#     mass = variables[2]
#     derivative = jnp.array([x[1],
#                            -kappa*x[0]/mass + 1.2*jnp.cos(0.628*t)])
#     return derivative

# @jit
# def hybrid_ode_stim(x, t, variables, parameters):
#     '''Calculates the right hand side of the hybrid ODE, where
#     the damping term is replaced by the neural network'''
#     kappa = variables[0]
#     mu = variables[1]
#     mass = variables[2]
#     derivative = jnp.array([jnp.array((x[1],)),
#                             jnp.array((-kappa*x[0]/mass,)) + jitted_neural_network(parameters, x) + jnp.array(1.2*jnp.cos(0.628*t))] ).flatten()
#     return derivative

# @jit
# def ode_aug(x, t, variables):
#     '''Calculates the right hand side of the original ODE.'''
#     kappa = variables[0]
#     mu = variables[1]
#     mass = variables[2]
#     derivative = [x[1],
#                  -kappa*x[0]/mass + (mu*(1-x[0]**2)*x[1])/mass]
#     derivative += [x[i] for i in range(2, x.shape[0])]

#     return jnp.array(derivative)

# @jit
# def hybrid_ode_aug(x, t, variables, parameters):
#     '''Calculates the right hand side of the hybrid ODE, where
#     the damping term is replaced by the neural network'''
#     kappa = variables[0]
#     mu = variables[1]
#     mass = variables[2]
#     derivative = [jnp.array((x[1],)),
#                   jnp.array((-kappa*x[0]/mass,)) + jitted_neural_network(parameters, x)]
#     derivative += [[x[i]] for i in range(2, x.shape[0])]
#     derivative = jnp.array(derivative).flatten()
#     return derivative

# @jit
# def ode_simple(x, t, variables):
#     '''Calculates the right hand side of the original ODE.'''
#     kappa = variables[0]
#     mu = variables[1]
#     mass = variables[2]
#     derivative = jnp.array([-0.1 * x[0] - 1 * x[1],
#                             1 * x[0] - 0.1 * x[1]])
#     return derivative

# @jit
# def hybrid_ode_simple(x, t, variables, parameters):
#     '''Calculates the right hand side of the hybrid ODE, where
#     the damping term is replaced by the neural network'''
#     kappa = variables[0]
#     mu = variables[1]
#     mass = variables[2]
#     derivative = jnp.array([jnp.array((-0.1 * x[0] - 1 * x[1],)),
#                             jitted_neural_network(parameters, x)]).flatten()
#     return derivative


def adjoint_f(adj, x, x_ref, t, variables, parameters, hybrid_ode, model_function):
    '''Calculates the right hand side of the adjoint system.'''
    df_dx = jax.jacobian(hybrid_ode, argnums=0)(x, t, variables, parameters, model_function)
    dg_dx = jax.grad(g, argnums=0)(x, x_ref, parameters)
    d_adj = - df_dx.T @ adj - dg_dx
    return d_adj

@jit
def adjoint_fmu(adj, x, x_ref, t, parameters, df_dx_at_t):
    '''Calculates the right hand side of the adjoint system.'''
    dg_dx = jax.grad(g, argnums=0)(x, x_ref, parameters)
    d_adj = - df_dx_at_t.T @ adj - dg_dx
    return d_adj


def g(x, x_ref, parameters):
    '''Calculates the inner part of the loss function.

    This function can either take individual floats for x
    and x_ref or whole numpy arrays'''
    return jnp.mean(0.5 * (x_ref - x)**2, axis = 0)

@jit
def J(x, x_ref, parameters):
    '''Calculates the complete loss of a trajectory w.r.t. a reference trajectory'''
    return jnp.mean(g(x, x_ref, parameters))


def create_residuals(x_ref, t, variables):
    x_dot = (x_ref[1:] - x_ref[:-1])/(t[1:] - t[:-1]).reshape(-1,1)
    v_ode = jax.vmap(lambda x_ref, t, variables: ode_res(x_ref, t, variables), in_axes=(0, 0, None))
    residual = x_dot - v_ode(x_ref[:-1], t[:-1], variables)
    return residual


def f_euler(x0, t, variables):
    '''Applies forward Euler to the original ODE and returns the trajectory'''
    x = jnp.zeros((t.shape[0], x0.shape[0]))
    x = x.at[0].set(x0)
    i = jnp.asarray(range(t.shape[0]))
    euler_body_func = partial(f_step, t=t, variables = variables)
    final, result = lax.scan(euler_body_func, x0, i)
    x = x.at[1:].set(result[:-1])
    return x

def f_step(prev_x, i, t, variables):
    t = jnp.asarray(t)
    dt = t[i+1] - t[i]
    next_x = prev_x + dt * ode(prev_x, t[i], variables)
    return next_x, next_x

def hybrid_euler(x0, t, variables, parameters, hybrid_ode, model_function):
    '''Applies forward Euler to the hybrid ODE and returns the trajectory'''
    x = jnp.zeros((t.shape[0], x0.shape[0]))
    x = x.at[0].set(x0)
    i = jnp.asarray(range(t.shape[0]))
    # We can replace the loop over the time by a lax.scan this is 3 times as fast: 0.32-0.26 -> 0.11-0.9
    euler_body_func = partial(hybrid_step, t=t, variables=variables, parameters=parameters, hybrid_ode=hybrid_ode, model_function=model_function)
    final, result = lax.scan(euler_body_func, x0, i)
    x = x.at[1:].set(result[:-1])
    # for i in range(len(t)-1):
    #     dt = t[i+1] - t[i]
    #     x[i+1] = x[i] + dt * hybrid_ode(x[i], t[i], variables, parameters)
    return x

def hybrid_step(prev_x, i, t, variables, parameters, hybrid_ode, model_function):
    t = jnp.asarray(t)
    dt = t[i+1] - t[i]
    next_x = prev_x + dt * hybrid_ode(prev_x, t[i], variables, parameters, model_function)
    return next_x, next_x

def adjoint_euler(a0, x, x_ref, t, variables, parameters, hybrid_ode, model_function):
    '''Applies forward Euler to the adjoint ODE and returns the trajectory'''
    a = jnp.zeros((t.shape[0], a0.shape[0]))
    a = a.at[0].set(a0)
    i = jnp.asarray(range(t.shape[0]))
    # We can replace the loop over the time by a lax.scan this is 3 times as fast: 0.32-0.26 -> 0.11-0.9
    euler_body_func = partial(adjoint_step, x=x, x_ref=x_ref, t=t, variables=variables,parameters=parameters, hybrid_ode=hybrid_ode, model_function=model_function)
    final, result = lax.scan(euler_body_func, a0, i)
    a = a.at[1:].set(result[:-1])
    return a

def adjoint_step(prev_a, i, x, x_ref, t, variables, parameters, hybrid_ode, model_function):
    t = jnp.asarray(t)
    x = jnp.asarray(x)
    x_ref = jnp.asarray(x_ref)
    dt = t[i+1]-t[i]
    next_a = prev_a + dt * adjoint_f(prev_a, x[i], x_ref[i], t[i], variables, parameters, hybrid_ode, model_function)
    return next_a, next_a

def adjoint_euler_fmu(a0, z, z_ref, t, optimisation_parameters, df_dz_trajectory):
    '''Applies forward Euler to the adjoint ODE and returns the trajectory'''
    a = np.zeros((t.shape[0], 2))
    a[0] = a0
    for i in range(len(t)-1):
        dt = t[i+1] - t[i] # nicht langsam
        d_adj = adjoint_fmu(a[i], z[i], z_ref[i], t[i], optimisation_parameters, df_dz_trajectory[i]) # lang, aber die steps brauchen nicht lang?
        a[i+1] = a[i] + dt * d_adj
    return a

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

def model_losses(x0, a0s, ts, variables, parameters, targets, hybrid_ode, model_function):
    xs = []
    losses = []
    adjoints = []
    # Compute Predictions
    for i, ic in enumerate(x0):
        x = hybrid_euler(ic, ts[i], variables, parameters, hybrid_ode, model_function)
        adjoint = adjoint_euler(a0s[i], np.flip(x, axis=0), np.flip(targets[i], axis=0), np.flip(ts[i]), variables, parameters, hybrid_ode, model_function)
        adjoint = np.flip(adjoint, axis=0)
        losses.append(float(J(x, targets[i], parameters)))
        xs.append(x)
        adjoints.append(adjoint)

    return xs, adjoints, losses


########################################################################################
#### JAX FUNCTIONS FMU  ################################################################
########################################################################################

df_dx_function_FMU = lambda dfmu_dz, dinput_dz, dfmu_dinput: dfmu_dz + dinput_dz * dfmu_dinput
vectorized_df_dx_function = jax.jit(jax.vmap(df_dx_function_FMU, in_axes=(0,0,0)))

########################################################################################
#### JAX FUNCTIONS COMMON  #############################################################
########################################################################################

dg_dtheta_function = lambda x, x_ref, parameters: jax.grad(g, argnums=2)(x, x_ref, parameters)
vectorized_dg_dtheta_function = jit(jax.vmap(dg_dtheta_function, in_axes=(0, 0, None)))

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
                x = nn.relu(x)
        return x

def create_nn(layers, x0):
    key1, key2, = random.split(random.PRNGKey(np.random.randint(0,100)), 2)
    neural_network = ExplicitMLP(features=layers)
    parameters = neural_network.init(key2, np.zeros((1, x0.shape[0])))
    jitted_neural_network = jax.jit(lambda params, inputs: neural_network.apply(params, inputs))
    return jitted_neural_network, parameters, neural_network

def residual_wrapper(parameters, args):
    pD = args['problemDescription']
    mD = args['modelDescription']
    oD = args['optimizerDescription']
    gS = args['generalSettings']
    residual = args['residual']
    restore = gS['load_parameters']
    reference_data = args['reference_data']
    pointers = args['pointers']
    results_directory = args['results_directory']
    logger = args['logger']
    model = args['model']
    unravel_pytree = args['unravel_pytree']

    J_residual = args['J_residual']
    checkpoint_manager = args['checkpoint_manager']

    t_train = reference_data['t_train']
    t_test = reference_data['t_test']
    x_ref_train = reference_data['x_ref_train']
    x_ref_test = reference_data['x_ref_test']

    start = time.time()

    parameters = unravel_pytree(parameters)

    # outputs = create_residuals(x_ref_train, t_train, pD['variables'])[:,1]
    # inputs = x_ref_train[:-1]
    if pD['fmu']:
        x_pred = args['fmu_evaluator'].euler(x_ref_train[0], t_train, model, parameters)
        args['fmu_evaluator'].reset_fmu(t_test[0], t_test[-1])
        x_pred_test = args['fmu_evaluator'].euler(x_ref_test[0], t_test, model, parameters)
        args['fmu_evaluator'].reset_fmu(t_train[0], t_train[-1])
    else:
        hybrid_ode = args['ode_hybrid']
        x_pred = hybrid_euler(x_ref_train[0], t_train, pD['variables'], parameters, hybrid_ode, model)
        x_pred_test = hybrid_euler(x_ref_test[0], t_test, pD['variables'], parameters, hybrid_ode, model)
    loss = float(J(x_pred, x_ref_train, parameters))
    loss_test = float(J(x_pred_test, x_ref_test, parameters))

    if oD['batching']:
        losses_res = []
        flat_gradients = []
        batch_preds = []
        for batch_idx, (input_train, output_train) in enumerate(args['train_dataloader']):
            input_train, output_train = input_train.to(gS['device']), output_train.to(gS['device'])
            input_train = input_train.detach().numpy()
            output_train = output_train.detach().numpy()
            (loss_res, batch_pred), gradient = jax.value_and_grad(J_residual, argnums=2, has_aux=True)(input_train, output_train, parameters)
            flat_gradient, _ = flatten_util.ravel_pytree(gradient)

            if args['epoch'] % gS['eval_freq'] == 0:
                if pD['fmu']:
                    t_batch = t_train[input_train.shape[0]*batch_idx:input_train.shape[0]*(batch_idx+1)]
                    args['fmu_evaluator'].reset_fmu(t_batch[0], t_batch[-1])
                    x_pred_batch = args['fmu_evaluator'].euler(x_ref_train[input_train.shape[0]*batch_idx], t_batch, model, parameters)
                else:
                    x_pred_batch = hybrid_euler(x_ref_train[input_train.shape[0]*batch_idx], t_train[input_train.shape[0]*batch_idx:input_train.shape[0]*(batch_idx+1)], pD['variables'], parameters, hybrid_ode, model)
                batch_preds.append(x_pred_batch)

            losses_res.append(loss_res)
            flat_gradients.append(flat_gradient)

            if batch_idx % 10 == 0:
                logger.info('Residual Training - Epoch: {}/{} [{}/{} ({:.0f}%)]\t Batch Loss: {:.6f}'.format(
                    args['epoch'], args['epochs'], batch_idx * len(input_train), len(args['train_dataloader'].dataset),
                    100. * batch_idx / len(args['train_dataloader']), loss_res))

        loss_res_batches = np.array(losses_res).sum()
        loss_res = loss_res_batches
        flat_gradient = np.mean(np.array(flat_gradients), 0)
        if args['epoch'] % gS['eval_freq'] == 0:
            x_pred_batch = np.vstack(batch_preds)
    else:
        inputs_train = args['train_dataloader'].dataset.x.detach().numpy()
        outputs_train = args['train_dataloader'].dataset.y.detach().numpy()
        (loss_res, pred), gradient = jax.value_and_grad(J_residual, argnums=2, has_aux=True)(inputs_train, outputs_train, parameters, model)
        flat_gradient, _ = flatten_util.ravel_pytree(gradient)
        loss_res_batches = None
        x_pred_batch = None

    args['losses_train'].append(loss)
    args['losses_train_res'].append(loss_res)
    args['losses_test'].append(loss_test)
    args['accuracies_train'].append(0.0)
    args['accuracies_test'].append(0.0)
    args['losses_batches'].append(loss_res_batches)

    if args['losses_test'][-1] < args['best_loss_test']:
        args['best_loss'] = args['losses_train'][-1]
        args['best_loss_test'] = args['losses_test'][-1]
        args['best_parameters'] = parameters
        args['best_pred'] = x_pred
        args['best_pred_test'] = x_pred_test
        if gS['save_parameters']:
            save_args = orbax_utils.save_args_from_target(parameters)
            checkpoint_manager.save(args['epoch'], parameters, save_kwargs={'save_args': save_args})

    if args['epoch'] % gS['eval_freq'] == 0:
        if gS['plot_prediction']:
            result_plot_multi_dim(mD['name'], pD['name'], os.path.join(results_directory, f'Prediction Epoch {args["epoch"]}.png'),
                            t_train, x_pred, t_test, x_pred_test,
                            np.hstack((t_train, t_test)), np.vstack((x_ref_train, x_ref_test)))
            if oD['batching']:
                result_plot_multi_dim(mD['name'], pD['name'], os.path.join(results_directory, f'Prediction (batched) Epoch {args["epoch"]}.png'),
                                t_train[:x_pred_batch.shape[0]], x_pred_batch, t_test, x_pred_test,
                                np.hstack((t_train, t_test)), np.vstack((x_ref_train, x_ref_test)))
            visualise_wb(parameters, results_directory, f'Parameters Epoch {args["epoch"]}')
        if gS['plot_loss']:
            build_plot(epochs=len(args['losses_train']),
                model_name=mD['name'],
                dataset_name=pD['name'],
                plot_path=os.path.join(results_directory, 'Loss.png'),
                train_acc=args['accuracies_train'],
                test_acc=args['accuracies_test'],
                train_loss=args['losses_train'],
                test_loss=args['losses_test']
            )
            if oD['batching']:
                build_plot(epochs=len(args['losses_train']),
                    model_name=mD['name'],
                    dataset_name=pD['name'],
                    plot_path=os.path.join(results_directory, 'Loss (batched).png'),
                    train_acc=args['accuracies_train'],
                    test_acc=args['accuracies_test'],
                    train_loss=args['losses_batches'],
                    test_loss=args['losses_test']
                )

    L2_regularisation = oD['lambda']/(2*(t_train.shape[0])) * np.linalg.norm(flat_gradient, 2)**2

    loss_cutoff = 1e-5
    if loss_res < loss_cutoff:
        warnings.warn('Terminating Optimization: Required Loss reached')

    epoch_time = time.time() - start

    logger.info('Epoch: {:04d}/{}    Loss (TrajTrain): {:.5e}    Loss (TrajTest): {:.5e}    Loss (ResTrain{}): {:.5e}'.format(
                    args['epoch'],
                    args['epochs'],
                    loss,
                    loss_test,
                    ', batched' if oD['batching'] else '',
                    loss_res
                    ))

    args['epoch'] += 1

    return loss_res + L2_regularisation, flat_gradient

def trajectory_wrapper(parameters, args):
    '''This is a function wrapper for the optimisation function. It returns the
    loss and the jacobian'''

    start = time.time()

    # Unpack the arguments
    pD = args['problemDescription']
    mD = args['modelDescription']
    oD = args['optimizerDescription']
    gS = args['generalSettings']

    residual = args['residual']
    restore = gS['load_parameters']
    reference_data = args['reference_data']
    pointers = args['pointers']
    results_directory = args['results_directory']
    epoch = args['epoch']
    epochs = args['epochs']
    logger = args['logger']

    model = args['model']
    unravel_pytree = args['unravel_pytree']
    checkpoint_manager = args['checkpoint_manager']

    t_train = reference_data['t_train']
    t_test = reference_data['t_test']
    x_ref_train = reference_data['x_ref_train']
    x_ref_test = reference_data['x_ref_test']

    start = time.time()

    parameters = unravel_pytree(parameters)

    if pD['fmu']:
        fmu_evaluator = args['fmu_evaluator']
        vectorized_dinput_dx_function = args['vectorized_dinput_dx_function_FMU']
        dinput_dtheta_function = args['dinput_dtheta_function_FMU']
        vectorized_df_dtheta_function = args['vectorized_df_dtheta_function_FMU']

        fmu_evaluator.training = False
        x_pred = fmu_evaluator.euler(x_ref_train[0], t_train, model, parameters)
        fmu_evaluator.reset_fmu(t_test[0], t_test[-1])
        x_pred_test = fmu_evaluator.euler(x_ref_test[0], t_test, model, parameters)
        fmu_evaluator.reset_fmu(t_train[0], t_train[-1])
        fmu_evaluator.training = True
    else:
        vectorized_df_dtheta_function = args['vectorized_df_dtheta_function']
        df_dt_function = args['df_dt_function']
        hybrid_ode = args['ode_hybrid']
        x_pred = hybrid_euler(x_ref_train[0], t_train, pD['variables'], parameters, hybrid_ode, model)
        x_pred_test = hybrid_euler(x_ref_test[0], t_test, pD['variables'], parameters, hybrid_ode, model)
    loss = float(J(x_pred, x_ref_train, parameters))
    loss_test = float(J(x_pred_test, x_ref_test, parameters))

    if oD['batching']:
        if pD['fmu']:
            pred_batches = []
            adjoints = []
            losses = []
            times = []
            gradients = []
            for batch_idx, (input_train, output_train) in enumerate(zip(args['inputs_train'], args['outputs_train'])):
                time_batch = input_train
                x_ref_train_batch = output_train
                x_pred_batch, dfmu_dz_batch, dfmu_dinput_batch = fmu_evaluator.euler(x_ref_train_batch[0], time_batch,model, parameters) # 0.06-0.09 sec
                dinput_dz_batch = vectorized_dinput_dx_function(parameters, x_pred_batch)
                df_dx_batch = vectorized_df_dx_function(dfmu_dz_batch, dinput_dz_batch, dfmu_dinput_batch)
                a0 = np.array([0, 0])
                adjoint_batch = adjoint_euler_fmu(a0, np.flip(x_pred_batch, axis=0), np.flip(x_ref_train_batch, axis=0), np.flip(time_batch), parameters, np.flip(np.asarray(df_dx_batch), axis=0)) # 0.025-0.035
                adjoint_batch = np.flip(adjoint_batch, axis=0)
                loss_batch = J(x_pred_batch, x_ref_train_batch, parameters)
                dinput_dtheta_batch = dinput_dtheta_function(parameters, x_pred_batch)

                df_dtheta_batch = vectorized_df_dtheta_function(dfmu_dinput_batch, dinput_dtheta_batch)
                dg_dtheta_batch = unfreeze(vectorized_dg_dtheta_function(x_pred_batch, x_ref_train_batch, parameters))

                for layer in df_dtheta_batch['params']:
                    # Sum the matmul result over the entire time_span to get the final gradients
                    df_dtheta_batch['params'][layer]['bias'] = np.einsum("iN,iNj->j", adjoint_batch, df_dtheta_batch['params'][layer]['bias'])
                    df_dtheta_batch['params'][layer]['kernel'] = np.einsum("iN,iNjk->jk", adjoint_batch, df_dtheta_batch['params'][layer]['kernel'])
                    dg_dtheta_batch['params'][layer]['kernel'] = np.einsum("Nij->ij", dg_dtheta_batch['params'][layer]['kernel'])
                    dg_dtheta_batch['params'][layer]['bias'] = np.einsum("Nj->j", dg_dtheta_batch['params'][layer]['bias'])

                pred_batches.append(x_pred_batch)
                # adjoints.append(adjoint)
                losses.append(loss_batch)
                times.append(time_batch)
                dJ_dtheta = jax.tree_map(lambda x, y: x+y, df_dtheta_batch, dg_dtheta_batch)
                flat_dJ_dtheta, _ = flatten_util.ravel_pytree(dJ_dtheta)
                gradients.append(flat_dJ_dtheta)

            loss_batches = np.asarray(losses).sum()
            accuracies_batches = 0.0
            pred_batched = np.vstack(pred_batches)
            gradient = np.array(gradients).sum(0)
        else:
            times = np.array(args['inputs_train'])
            x_ref_train_batches = np.array(args['outputs_train'])
            a0s = np.zeros((len(args['inputs_train']), *x_ref_train[0].shape))
            x0s = x_ref_train_batches[:,0]
            pred_batches, adjoints, losses = model_losses(x0s, a0s, times, pD['variables'], parameters, x_ref_train_batches, hybrid_ode, model)

            loss_batches = np.asarray(losses).sum()
            accuracies_batches = 0.0
            gradients = []
            for x_, adjoint_, t_ in zip(pred_batches, adjoints, times):
                # Calculate the gradient of the hybrid ode with respect to the parameters
                df_dtheta_trajectory = vectorized_df_dtheta_function(x_, t_, pD['variables'], parameters)
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
            gradient = np.asarray(gradients).sum(0)
            pred_batched = np.vstack(pred_batches)
    else:
        a0 = np.zeros(x_ref_train[0].shape)
        adjoint = adjoint_euler(a0, np.flip(x_pred, axis=0), np.flip(x_ref_train, axis=0), np.flip(t_train), pD['variables'], parameters, hybrid_ode, model)
        adjoint = np.flip(adjoint, axis=0)
        # Calculate the gradient of the hybrid ode with respect to the parameters
        df_dtheta_trajectory = vectorized_df_dtheta_function(x_ref_train, t_train, pD['variables'], parameters)
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
        loss_batches = loss
        accuracies_batches = 0.0
        pred_batched = x_pred
        gradient = flat_dJ_dtheta

    # if random_shift:
    #     if np.abs(loss - losses[-1]) < 0.1:
    #         flat_dJ_dtheta += np.random.normal(0, np.linalg.norm(flat_dJ_dtheta,2), flat_dJ_dtheta.shape)


    args['losses_train'].append(loss)
    args['losses_test'].append(loss_test)
    args['accuracies_train'].append(0.0)
    args['accuracies_test'].append(0.0)
    args['losses_batches'].append(loss_batches)

    if args['losses_test'][-1] < args['best_loss']:
        args['best_loss'] = args['losses_train'][-1]
        args['best_loss_test'] = args['losses_test'][-1]
        args['best_parameters'] = parameters
        args['best_pred'] = x_pred
        args['best_pred_test'] = x_pred_test
        if gS['save_parameters']:
            save_args = orbax_utils.save_args_from_target(parameters)
            checkpoint_manager.save(epoch, parameters, save_kwargs={'save_args': save_args})

    if epoch % gS['eval_freq'] == 0:
        if gS['plot_prediction']:
            result_plot_multi_dim(mD['name'], pD['name'], os.path.join(results_directory, f'Prediction Epoch {args["epoch"]}.png'),
                            t_train, x_pred, t_test, x_pred_test,
                            np.hstack((t_train, t_test)), np.vstack((x_ref_train, x_ref_test)))
            if oD['batching']:
                result_plot_multi_dim(mD['name'], pD['name'], os.path.join(results_directory, f'Prediction (batched) Epoch {args["epoch"]}.png'),
                            t_train[:pred_batched.shape[0]], pred_batched, t_test, x_pred_test,
                            np.hstack((t_train, t_test)), np.vstack((x_ref_train, x_ref_test)))
        if gS['plot_loss']:
                build_plot(epochs=len(args['losses_train']),
                    model_name=mD['name'],
                    dataset_name=pD['name'],
                    plot_path=os.path.join(results_directory, 'Loss.png'),
                    train_acc=args['accuracies_train'],
                    test_acc=args['accuracies_test'],
                    train_loss=args['losses_train'],
                    test_loss=args['losses_test']
                )
                if oD['batching']:
                    build_plot(epochs=len(args['losses_batches']),
                        model_name=mD['name'],
                        dataset_name=pD['name'],
                        plot_path=os.path.join(results_directory, 'Loss (batched).png'),
                        train_acc=args['accuracies_train'],
                        test_acc=args['accuracies_test'],
                        train_loss=args['losses_batches'],
                        test_loss=args['losses_test']
                    )
        if gS['plot_parameters']:
            visualise_wb(parameters, results_directory, f'Parameters Epoch {args["epoch"]}.png')

    L2_regularisation = oD['lambda']/(2*(t_train.shape[0])) * np.linalg.norm(gradient, 2)**2

    end = time.time()

    # loss_cutoff = 1e-5
    # if loss_res < loss_cutoff:
    #     warnings.warn('Terminating Optimization: Required Loss reached')

    logger.info('Epoch: {:04d}/{}    Loss (TrajTrain): {:.5e}    Loss (TrajTest): {:.5e}    Loss (TrajTrain{}): {:.5e}'.format(
                    epoch,
                    args['epochs'],
                    loss,
                    loss_test,
                    ', batched' if oD['batching'] else '',
                    loss_batches
                    ))
    args['epoch'] += 1
    return loss + L2_regularisation, gradient


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--kappa', type=float, default=1.0, help='oscilation constant of the VdP osc. term')
    parser.add_argument('--mu', type=float, default=8.53, help='damping value of the VdP damping term')
    parser.add_argument('--mass', type=float, default=1.0, help='mass of the VdP system')

    parser.add_argument('--start', type=float, default=0.0, help='Start value of the ODE integration')
    parser.add_argument('--end', type=float, default=20.0, help='End value of the ODE integration')
    parser.add_argument('--n_steps', type=int, default=1001, help='How many integration steps to perform')

    parser.add_argument('--layers', type=int, default=1, help='Number of hidden layers')
    parser.add_argument('--layer_size', type=int, default=15, help='Number of neurons in a hidden layer')
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

    parser.add_argument('--method', type=str, default='BFGS', help='Which optimisation method to use')
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

    if args.aug_state:
        ode = ode_aug
        hybrid_ode = hybrid_ode_aug
        x0 = [1.0, 0.0]
        x0 += [0.0 for i in range(args.aug_dim)]
        x0 = np.asarray(x0)
    elif args.stimulate:
        ode = ode_stim
        hybrid_ode = hybrid_ode_stim
        ode_res = ode_stim_res
        x0 = np.array([1.0, 0.0])
    elif args.simple_problem:
        ode = ode_simple
        hybrid_ode = hybrid_ode_simple
        x0 = np.array([2.0, 0.0])
    else:
        x0 = np.array([1.0, 0.0])

    t_ref = np.linspace(args.start, args.end, args.n_steps)
    t_val = np.linspace(args.end, (args.end-args.start) * 1.5, int(args.n_steps * 0.5))
    reference_variables = np.asarray([args.kappa, args.mu, args.mass])
    x_ref = f_euler(x0, t_ref, reference_variables)
    x0_val = x_ref[-1]
    x_val = f_euler(x0_val, t_val, reference_variables)

    layers = [args.layer_size]*args.layers
    layers.append(1)
    jitted_neural_network, parameters = create_nn(layers, x0)

    if args.restore:
        # Restore previous parameters
        step = checkpoint_manager.latest_step()
        parameters = checkpoint_manager.restore(step)

    flat_parameters, unravel_pytree = flatten_util.ravel_pytree(parameters)

    epoch = 0

    # Put all arguments the optimization needs into one array for the minimize function
    args = {'time' : t_ref,
                    'val_time': t_val,
                    'initial_condition' : x0,
                    'reference_solution' : x_ref,
                    'validation_solution' : x_val,
                    'reference_variables' : reference_variables,
                    'unravel_pytree' : unravel_pytree,
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
                    'saved_parameters' : parameters,
                    'best_loss' : np.inf,
                    'best_loss_val': np.inf,
                    'checkpoint_manager' : checkpoint_manager,
                    'lambda' : args.lambda_,
                    'logger': logger}

    if args.clean_batching:
        x0s, targets, ts = create_clean_mini_batch(args.clean_n_batches, x_ref, t_ref)
        args['x0s'] = x0s
        args['targets'] = targets
        args['ts'] = ts
    else:
        args['x0s'] = None
        args['targets'] = None
        args['ts'] = None

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
                    loss, grad = residual_wrapper(flat_parameters=flat_parameters, args=args)
                    flat_parameters = Adam.update(i+1, flat_parameters, grad)
            else:
                residual_result = minimize(residual_wrapper, flat_parameters, method=args.method, jac=True, args=args, options={'maxiter':args.res_steps})
        except (KeyboardInterrupt, UserWarning):
            pass
        experiment_time = time.time() - start
        parameters = args['saved_parameters']
        best_loss = args['best_loss']
        logger.info(f'Best Loss in Residual Training: {best_loss}')
        flat_parameters, _ = flatten_util.ravel_pytree(parameters)
        # Plot the result of the training on residuals
        x = hybrid_euler(x0, t_ref, reference_variables, parameters)
        plot_results(t_ref, x, x_ref, residual_path+'_best')
        plot_losses(epochs=list(range(len(args['losses']))), training_losses=args['losses'], validation_losses=args['losses_val'], path=residual_path+'_losses')
        plot_losses(epochs=list(range(len(args['losses_res']))), training_losses=args['losses_res'], path=residual_path+'_losses_res')

        results_dict['losses_train_res'] =  list(args['losses_res'])
        results_dict['time_res'] = experiment_time

    # Train on Trajectory
    ####################################################################################
    args['results_path'] = trajectory_path
    start = time.time()
    try:
        if args.method == 'Adam':
            Adam = AdamOptim(eta=args.adam_eta*10, beta1=args.adam_beta1, beta2=args.adam_beta2, epsilon=args.adam_eps)
            for i in range(args.opt_steps):
                loss, grad = function_wrapper(flat_parameters=flat_parameters, args=args)
                flat_parameters = Adam.update(i+1, flat_parameters, grad)
        else:
            res = minimize(function_wrapper, flat_parameters, method=args.method, jac=True, args=args, tol=args.tol)
    except (KeyboardInterrupt, UserWarning):
        pass

    experiment_time = time.time() - start

    results_dict['losses_train_traj'] =  list(args['losses'])
    results_dict['losses_train_traj_val'] = list(args['losses_val'])
    results_dict['time_traj'] = experiment_time
    logger.info(f'Dumping results to {args.results_file}.')
    with open(args.results_file, 'w') as file:
        yaml.dump(results_dict, file)

    parameters = args['saved_parameters']
    flat_parameters, _ = flatten_util.ravel_pytree(parameters)
    dumpable_params = []
    for param in flat_parameters:
        dumpable_params.append(float(param))
    with open(args.results_file, 'a') as file:
        yaml.dump({'flat_parameters': dumpable_params}, file)

    best_loss = args['best_loss']
    logger.info(f'Best Loss in Training: {best_loss}')

    z_training = hybrid_euler(x0, t_ref, reference_variables, parameters)
    x_validation = hybrid_euler(x0_val, t_val, reference_variables, parameters)

    plot_results(t_ref, z_training, x_ref, trajectory_path+'_best')
    plot_results(t_val, x_validation, x_val, trajectory_path+'_best_val')
    plot_losses(epochs=range(len(args['losses'])), training_losses=args['losses'], validation_losses=args['losses_val'], path=trajectory_path)


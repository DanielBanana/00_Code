import os
import sys
import shutil
# import torch
# import torch.nn.functional as F
import numpy as np
import argparse
import time
import yaml
import matplotlib.pyplot as plt


import jax
from jax import lax, jit, flatten_util
import jax.numpy as jnp
import flax.linen as nn
# this only works on startup!
from jax.config import config
config.update("jax_enable_x64", True)
from functools import partial

sys.path.insert(1, os.getcwd())
from plot_results import plot_results, plot_losses, get_file_path
from fmu_helper import FMUEvaluator
from cbo_in_python.src.jax_.models import *
# from cbo_in_python.src.datasets import load_mnist_dataloaders, load_parabola_dataloaders, f
from cbo_in_python.src.datasets import create_generic_dataset, load_generic_dataloaders
from cbo_in_python.src.jax_.optimizer import Optimizer
from cbo_in_python.src.jax_.loss import Loss
from torch.utils.data import Dataset, DataLoader

from collections import OrderedDict
from pyDOE2 import fullfact
import datetime

import orbax
from flax.training import orbax_utils

from utils import build_plot, result_plot, create_results_directory, create_doe_experiments, create_experiment_directory

MODELS = {
    # 'SimpleMLP': SimpleMLP,
    'TinyMLP': TinyMLP,
    'SmallMLP': SmallMLP,
    # 'LeNet1': LeNet1,
    # 'LeNet5': LeNet5,
    'CustomMLP': CustomMLP
}

DATASETS = {
    'VdP': ''
}

# # The Neural Network structure class
# class VdPMLP(nn.Module):
#     def __init__(self):
#         super(VdPMLP, self).__init__()

#         self.model = nn.Sequential(
#             nn.Linear(2, 20),
#             nn.ReLU(),
#             nn.Linear(20, 20),
#             nn.ReLU(),

#             nn.Linear(20, 1)
#         )
#         self.double()

#     def forward(self, x):
#         return self.model(x)



# Managing class for Hybrid model with FMU
class Hybrid_FMU(nn.Module):
    def __init__(self, fmu_model, augment_model, z0, t, mode='trajectory'):
        super(Hybrid_FMU, self).__init__()

        self.fmu_model = fmu_model
        self.augment_model = augment_model
        self.z0 = z0
        self.t = t

        if mode=='residual':
            self.forward = self.forward_residual
        elif mode=='trajectory':
            self.forward = self.forward_trajectory
        else:
            print('Mode not recognised. Using residual mode')
            self.forward = self.forward_residual

    def residual_mode(self):
        self.forward = self.forward_residual

    def trajectory_mode(self):
        self.forward = self.forward_trajectory

    def set_trajectory_variables(self, z0, t):
        self.z0 = z0
        self.t = t

    # def augment_model_function(self, augment_model_parameters, input):
    #     # The augment_model is currently a pytorch model, which just takes
    #     # the input. It has its own parameters saved internally.
    #     # The f_euler function expects a model which needs its paramaters
    #     # given when it is called: y = augment_model_function(parameters, input)
    #     # f_euler provides the input to the augment_model as numpy array
    #     # but we can only except tensors, so convert
    #     return self.augment_model(torch.tensor(input)).detach().numpy()

    def forward_trajectory(self, pointers):
        '''Applies euler to the VdP ODE by calling the fmu; returns the trajectory'''
        t = self.t
        z0 = self.z0
        z = np.zeros((t.shape[0], 2))
        z[0] = z0
        # Forward the initial state to the FMU
        self.fmu_model.setup_initial_state(z0, pointers)
        times = []
        for i in range(len(t)-1):
            # start = time.time()
            status = self.fmu_model.fmu.setTime(t[i])
            dt = t[i+1] - t[i]

            pointers, enterEventMode, terminateSimulation = self.fmu_model.evaluate_fmu(t[i], dt, self.augment_model, None, pointers)

            z[i+1] = z[i] + dt * pointers.dx

            if terminateSimulation:
                break

        return z


# PYTHON ONLY ODEs
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

def create_residual_references(z_ref, t, ode_parameters):
    z_dot = (z_ref[1:] - z_ref[:-1])/(t[1:] - t[:-1]).reshape(-1,1)
    v_ode = jax.vmap(lambda z_ref, t, ode_parameters: ode_res(z_ref, t, ode_parameters), in_axes=(0, 0, None))
    residual = z_dot - v_ode(z_ref[:-1], t[:-1], ode_parameters)
    return residual

def J_residual(inputs, outputs, neural_network, nn_parameters):
    def squared_error(input, output, neural_network, nn_parameters):
        pred = neural_network(nn_parameters, input, neural_network)
        return (output-pred)**2
    error_function = partial(squared_error, neural_network=neural_network, nn_parameters=nn_parameters)
    return jnp.mean(jax.vmap(error_function)(inputs, outputs), axis=0)[0]

# Python only Hybrid model
class Hybrid_Python():
    def __init__(self, z0, t, ode_parameters, augment_model, nn_parameters, mode='trajectory', restore=False):

        self.ode_parameters = ode_parameters
        self.augment_model = augment_model
        self.z0 = z0
        self.t = t
        self.nn_parameters = nn_parameters
        _, self.unravel_pytree = jax.flatten_util.ravel_pytree(nn_parameters)
        self.ravel_pytree = jax.jit(jax.flatten_util.ravel_pytree)
        self.unravel_pytree = jax.jit(self.unravel_pytree)
        self.stim = False
        self.restore = restore

        if mode=='residual':
            self.residual_mode()
        elif mode=='trajectory':
            self.trajectory_mode()
        else:
            print('Mode not recognised. Using residual mode')
            self.residual_mode()

    def __call__(self, input=None):
        return self.forward(input)

    def residual_mode(self):
        self.forward = self.forward_residual

    def trajectory_mode(self):
        self.forward = self.forward_trajectory

    def set_trajectory_variables(self, z0, t):
        self.z0 = z0
        self.t = t

    def forward_residual(self, input):
        return self.augment_model(self.nn_parameters, input)

    def forward_trajectory(self, input):
        # Forward the initial state to the FMU
        t = input[:,0]
        z_ref = input[0,1:]
        times = []
        if self.stim:
            self.hybrid_ode = self.hybrid_ode_stim
        else:
            # NN parameters are saved inside the augment_model and therefore not needed as input
            self.hybrid_ode = self.hybrid_ode
        z = self.hybrid_euler(z0, t, ode_parameters=self.ode_parameters, nn_parameters=self.nn_parameters)
        return z

    def hybrid_euler(self, z0, t, ode_parameters, nn_parameters):
        '''Applies forward Euler to the hybrid ODE and returns the trajectory'''
        z = np.zeros((t.shape[0], z0.shape[0]))
        z[0] = z0
        i = np.asarray(range(t.shape[0]))
        # We can replace the loop over the time by a lax.scan this is 3 times as fast: e.g.: 0.32-0.26 -> 0.11-0.9
        # euler_body_func = partial(self.hybrid_step, t=t, ode_parameters=ode_parameters, nn_parameters=nn_parameters)
        # final, result = lax.scan(euler_body_func, z0, i)
        # z = z.at[1:].set(result[:-1])
        for i in range(len(t)-1):
            dt = t[i+1] - t[i]
            z[i+1] = z[i] + dt * self.hybrid_ode(z[i], t[i], ode_parameters, nn_parameters)
            # z[i+1] = z[i] + dt * (self.deficient_ode(z[i], t[i], ode_parameters) + np.array([0,1])*self.augment_model(nn_parameters, z[i]))
        return z

    @jit
    def deficient_ode(self, z, t, ode_parameters):
        kappa = ode_parameters[0]
        mu = ode_parameters[1]
        mass = ode_parameters[2]
        derivative = np.array([np.array((z[1],)),
                               np.array((-kappa*z[0]/mass,))]).flatten()
        return derivative


    def hybrid_ode(self, z, t, ode_parameters, nn_parameters):
        '''Calculates the right hand side of the hybrid ODE, where
        the damping term is replaced by the neural network'''
        kappa = ode_parameters[0]
        mu = ode_parameters[1]
        mass = ode_parameters[2]
        derivative = np.array([np.array((z[1],)),
                               np.array((-kappa*z[0]/mass,)) + self.augment_model(nn_parameters, z)]).flatten()
        return derivative

    def hybrid_ode_stim(self, z, t, ode_parameters, nn_parameters):
        '''Calculates the right hand side of the hybrid ODE, where
        the damping term is replaced by the neural network'''
        kappa = ode_parameters[0]
        mu = ode_parameters[1]
        mass = ode_parameters[2]
        derivative = np.array([np.array((z[1],)),
                                np.array((-kappa*z[0]/mass,)) + self.augment_model(nn_parameters, z) + np.array(1.2*np.cos(np.pi/5*t))]).flatten()
        return derivative

    def parameters(self):
        parameters, self.treedef = jax.tree_util.tree_flatten(self.nn_parameters)
        # a, b = tree_flatten(self.nn_parameters)
        # self.treedef = b
        return parameters

    # @jit
    def parameters_flat(self):
        parameters, _ = jax.flatten_util.ravel_pytree(self.nn_parameters)
        return parameters

    # @jit
    def set_parameters(self, parameters):
        self.nn_parameters = jax.tree_util.tree_unflatten(self.treedef, parameters)

    # @jit
    def set_parameters_flat(self, parameters):

        self.nn_parameters = self.unravel_pytree(parameters)
        # self.nn_parameters = self.unravel_pytree(parameters)

# For calculation of the reference solution we need the correct behaviour of the VdP
def damping(mu, inputs):
    return mu * (1 - inputs[0]**2) * inputs[1]

def _evaluate_reg(outputs, y_, loss_fn):
    # with torch.no_grad():
    #     loss = loss_fn(outputs, y_)
    # return loss
    loss = loss_fn(outputs, y_)
    return loss

def mse_loss(input: jnp.array, target: jnp.array):
    return jnp.mean((input-target)**2)

def train(model:Hybrid_FMU or Hybrid_Python,
          train_dataloader,
          test_dataloader,
          device,
          use_multiprocessing,
          processes,
          epochs,
          particles,
          particles_batch_size,
          alpha,
          sigma,
          l,
          dt,
          anisotropic,
          eps,
          partial_update,
          cooling,
          checkpoint_interval,
          problem_type,
          pointers,
          checkpoint_manager,
          residual=False,
          plotting_reference_data={}):

    accuracies_train = []
    losses_train = []
    accuracies_test = []
    losses_test = []
    best_loss = np.inf
    best_epoch = 0
    best_parameters = model.nn_parameters

    start_opt_setup = time.time()
    # Optimizes the Neural Network with CBO
    optimizer = Optimizer(model, n_particles=particles, alpha=alpha, sigma=sigma,
                          l=l, dt=dt, anisotropic=anisotropic, eps=eps, partial_update=partial_update,
                          use_multiprocessing=use_multiprocessing, n_processes=processes,
                          particles_batch_size=particles_batch_size, device=device, fmu=False, residual=residual)
    print(f'time for optimizer setup: {time.time()-start_opt_setup}')

    if problem_type == 'classification':
        raise NotImplementedError('No classification loss currently available')
        # loss_fn = Loss(F.nll_loss, optimizer)
    else:
        # loss_fn = Loss(F.mse_loss, augment_optimizer)
        loss_fn = Loss(mse_loss, optimizer)

    n_batches = len(train_dataloader)

    for epoch in range(epochs):
        start_epoch = time.time()
        accuracies_epoch_train = []
        losses_epoch_train = []
        time_train_evals = []
        time_optimizer_steps = []
        for batch, (input_train, output_train) in enumerate(train_dataloader):
            # input_train, output_train = input_train.to(device), output_train.to(device)
            input_train, output_train = np.array(input_train), np.array(output_train)

            # If we calculate on the residual the inputs are the states of the ODE
            # e.g. location and velocity for Van der Pol. The outputs are the residuals
            # of the ODE derivatives, i.e. the part of the derivative that is currently
            # unaccounted for by the simple ODE model
            if problem_type == 'classification':
                # loss_train, train_acc = _evaluate_class(model, X, y, F.nll_loss)
                raise NotImplementedError('Evaluation function for classification not implemented')

            if isinstance(model, Hybrid_FMU):
                pred_train = np.array(model(pointers))
            else:
                start_pred = time.time()
                pred_train = np.array(model(input_train))
                time_train_eval = time.time() - start_pred
                time_train_evals.append(time_train_eval)


            # loss_train = _evaluate_reg(pred_train, output_train, F.mse_loss).cpu()
            loss_train = _evaluate_reg(pred_train, output_train, mse_loss)
            train_acc = 0.0

            accuracies_epoch_train.append(train_acc)
            losses_epoch_train.append(loss_train)

            # optimizer.zero_grad()
            loss_fn.backward(input_train, output_train, backward_gradients=False)
            start = time.time()
            optimizer.step()
            time_optimizer_step = time.time() - start
            time_optimizer_steps.append(time_optimizer_step)

            # Evaluate test set
            # with torch.no_grad():
            losses = []
            accuracies = []
            for input_test, output_test in test_dataloader:
                # input_test, output_test = input_test.to(device), output_test.to(device)
                input_test, output_test = np.array(input_test), np.array(output_test)

                if problem_type == 'classification':

                    # loss, acc = _evaluate_class(model, X_test, y_test, F.nll_loss)
                    raise NotImplementedError('Evaluation function for classification not implemented')

                if isinstance(model, Hybrid_FMU):
                    pred_test = np.array(model(pointers))
                else:
                    pred_test = np.array(model(input_test))

                loss = _evaluate_reg(pred_test, output_test, mse_loss)
                acc = 0.0
                # losses.append(loss.cpu())
                losses.append(loss)
                accuracies.append(acc)
            loss_test, test_acc = np.mean(losses), np.mean(accuracies)
            if batch == n_batches - 1:
                accuracies_test.append(test_acc)
                losses_test.append(loss_test)

        accuracies_train.append(np.mean(accuracies_epoch_train))
        losses_train.append(np.mean(losses_epoch_train))

        if epoch > 0 and loss_test < best_loss:
            best_epoch = epoch
            best_loss = loss_test
            best_parameters = model.nn_parameters
            # save_args = orbax_utils.save_args_from_target(nn_parameters)
            # checkpoint_manager.save(epoch, nn_parameters, save_kwargs={'save_args': save_args})

        if epoch % checkpoint_interval == 0:
            input_trajectory_train = plotting_reference_data['input_trajectory_train']
            input_trajectory_test = plotting_reference_data['input_trajectory_test']
            results_directory = plotting_reference_data['plot_directory']

            model.trajectory_mode()

            if isinstance(model, Hybrid_FMU):
                pred_train = np.array(model(pointers))
            else:
                pred_train = model(input_trajectory_train)

            if isinstance(model, Hybrid_FMU):
                pred_test = np.array(model(pointers))
            else:
                pred_test = model(input_trajectory_test)

            if residual:
                model.residual_mode()

            result_plot('Custom', 'VdP', os.path.join(results_directory, f'Epoch {epoch}.png'),
                        t_train, pred_train, t_test, pred_test,
                        np.hstack((t_train, t_test)), np.vstack((z_ref_train, z_ref_test)))

        print(f'Epoch: {epoch + 1:2}/{epochs}, batch: {batch + 1:4}/{n_batches}, train loss: {losses_train[-1]:8.3f}, '
              f'train acc: {accuracies_train[-1]:8.3f}, test loss: {losses_test[-1]:8.3f}, test acc: {accuracies_test[-1]:8.3f}, alpha: {optimizer.alpha:8.3f}, sigma: {optimizer.sigma:8.6f}, dt: {optimizer.dt:8.6f}',
              flush=True)

        if cooling:
            optimizer.cooling_step()

        if np.isnan(losses_train[-1]):
            losses_train = losses_train + [np.inf]*(epochs-epoch-1)
            losses_test = losses_test + [np.inf]*(epochs-epoch-1)
            accuracies_train = accuracies_train + [np.inf]*(epochs-epoch-1)
            accuracies_test = accuracies_test + [np.inf]*(epochs-epoch-1)
            break

        time_epoch = time.time() - start_epoch
        time_optimizer_step_mean = np.mean(np.array(time_optimizer_steps))
        time_train_evals_mean = np.mean(np.array(time_train_evals))
        print(f'time for a full epoch: {time_epoch}')
        print(f'number of sample batches: {n_batches}')
        print(f'time for a optimizer step: {time_optimizer_step_mean}')
        print(f'time for a evluation of the model: {time_train_evals_mean}')



    return accuracies_train, accuracies_test, losses_train, losses_test, best_epoch, best_parameters

def f_euler_fmu(z0, t, fmu_evaluator: FMUEvaluator, model, model_parameters, pointers):
    '''Applies euler to the VdP ODE by calling the fmu; returns the trajectory'''
    z = np.zeros((t.shape[0], 2))
    z[0] = z0
    # Forward the initial state to the FMU
    fmu_evaluator.setup_initial_state(z0, pointers)
    times = []
    if fmu_evaluator.training:
        dfmu_dz_trajectory = []
        dfmu_dinput_trajectory = []
    for i in range(len(t)-1):
        # start = time.time()
        status = fmu_evaluator.fmu.setTime(t[i])
        dt = t[i+1] - t[i]

        pointers, enterEventMode, terminateSimulation = fmu_evaluator.evaluate_fmu(t[i], dt, model, model_parameters, pointers)

        z[i+1] = z[i] + dt * pointers.dx

        if terminateSimulation:
            break
    return z

def f_euler_python(z0, t, ode_parameters):
    '''Applies forward Euler to the original ODE and returns the trajectory'''
    z = jnp.zeros((t.shape[0], z0.shape[0]))
    z = z.at[0].set(z0)
    i = jnp.asarray(range(t.shape[0]))
    euler_body_func = partial(f_step, t=t, ode_parameters = ode_parameters)
    final, result = lax.scan(euler_body_func, z0, i)
    z = z.at[1:].set(result[:-1])
    return z

def f_step(prev_z, i, t, ode_parameters):
    t = jnp.asarray(t)
    dt = t[i+1] - t[i]
    next_z = prev_z + dt * ode(prev_z, t[i], ode_parameters)
    return next_z, next_z

# def build_plot(epochs, model_name, dataset_name, plot_path,
#                train_acc, test_acc, loss_train, loss_test):
#     plt.rcParams['figure.figsize'] = (20, 10)
#     plt.rcParams['font.size'] = 25

#     epochs_range = np.arange(1, epochs + 1, dtype=int)

#     # plt.clf()
#     fig, (ax1, ax2) = plt.subplots(1, 2)

#     ax1.plot(epochs_range, train_acc, label='train')
#     ax1.plot(epochs_range, test_acc, label='test')
#     ax1.legend()
#     ax1.set_xlabel('epoch')
#     ax1.set_ylabel('accuracy')
#     ax1.set_title('Accuracy')

#     ax2.plot(epochs_range, loss_train, label='train')
#     ax2.plot(epochs_range, loss_test, label='test')
#     ax2.legend()
#     ax2.set_xlabel('epoch')
#     ax2.set_ylabel('loss')
#     ax2.set_title('Loss')

#     fig.suptitle(f'{model_name} @ {dataset_name}')
#     fig.savefig(plot_path)
#     plt.close(fig)

# def result_plot(model_name, dataset_name, plot_path,
#                 input_train, output_train, input_test, output_test, input_reference, output_reference, scatter=False):
#     plt.rcParams['figure.figsize'] = (20, 10)
#     plt.rcParams['font.size'] = 25

#     # Get the dimensions of the output
#     output_dims = output_train.shape[1]

#     fig, axes = plt.subplots(output_dims, 2)

#     for out_dim in range(output_dims):

#         axes[out_dim, 0].plot(input_reference, output_reference[:,out_dim], label='Reference')
#         if scatter:
#             axes[out_dim, 0].scatter(input_train, output_train[:,out_dim], label='Prediction')
#         else:
#             axes[out_dim, 0].plot(input_train, output_train[:,out_dim], label='Prediction')
#         axes[out_dim, 0].legend()
#         axes[out_dim, 0].set_xlabel('X')
#         axes[out_dim, 0].set_ylabel('y')
#         axes[out_dim, 0].set_title(f'Variable {out_dim+1} - Train')

#         axes[out_dim, 1].plot(input_reference, output_reference[:,out_dim], label='Reference')
#         if scatter:
#             axes[out_dim, 1].scatter(input_test, output_test[:,out_dim], label='Prediction')
#         else:
#             axes[out_dim, 1].plot(input_test, output_test[:,out_dim], label='Prediction')
#         axes[out_dim, 1].legend()
#         axes[out_dim, 1].set_xlabel('X')
#         axes[out_dim, 1].set_ylabel('y')
#         axes[out_dim, 1].set_title(f'Variable {out_dim+1} - Test')

#     fig.tight_layout()
#     fig.suptitle(f'{model_name} @ {dataset_name}')
#     fig.savefig(plot_path)
#     plt.close(fig)

# def create_results_directory(directory, results_directory_name=None):
#     if results_directory_name is None:
#         now = datetime.datetime.now()
#         doe_date = '-'.join([str(now.year), str(now.month), str(now.day)]) + '_' + '-'.join([str(now.hour), str(now.minute)])
#         doe_directory = os.path.join(directory, doe_date)
#     else:
#         doe_directory = os.path.join(directory, results_directory_name)
#         if not os.path.exists(doe_directory):
#             os.mkdir(doe_directory)
#         else:
#             count = 1
#             while os.path.exists(doe_directory):
#                 doe_directory = os.path.join(directory, results_directory_name + f'_{count}')
#                 count += 1
#             os.mkdir(doe_directory)
#     return doe_directory

def create_results_subdirectories(results_directory, trajectory=False, residual=False, checkpoint=True):
    return_directories = []
    if trajectory:
        trajectory_directory = os.path.join(results_directory, 'trajectory')
        if not os.path.exists(trajectory_directory):
            os.mkdir(trajectory_directory)
        return_directories.append(trajectory_directory)

    if residual:
        residual_directory = os.path.join(results_directory, 'residual')
        if not os.path.exists(residual_directory):
            os.mkdir(residual_directory)
        return_directories.append(residual_directory)

    if checkpoint:
        checkpoint_directory = os.path.join(results_directory, 'ckpt')
        if not os.path.exists(checkpoint_directory):
            os.mkdir(checkpoint_directory)
        return_directories.append(checkpoint_directory)

    return tuple(return_directories)

def create_results_subdirectories_(results_directory):
    plots_directory = os.path.join(results_directory, 'plots')
    if not os.path.exists(plots_directory):
        os.mkdir(plots_directory)
    checkpoint_directory = os.path.join(results_directory, 'ckpt')
    if not os.path.exists(checkpoint_directory):
        os.mkdir(checkpoint_directory)
    return plots_directory, checkpoint_directory

def create_reference_solution(start, end, n_steps, z0, reference_ode_parameters, ode_integrator):

    t_train = np.linspace(start, end, n_steps)
    z_ref_train = np.asarray(ode_integrator(z0, t_train, reference_ode_parameters))

    # Generate the reference data for testing
    z0_test = z_ref_train[-1]
    n_steps_test = int(n_steps * 0.5)
    t_test = np.linspace(end, (end - start) * 1.5, n_steps_test)
    z_ref_test = np.asarray(ode_integrator(z0_test, t_test, reference_ode_parameters))

    return t_train, z_ref_train, t_test, z_ref_test, z0_test

def create_residual_reference_solution(t_train, z_ref_train, t_test, z_ref_test, reference_ode_parameters):

    # CREATE RESIDUALS FROM TRAJECTORIES
    output_residual_train = np.asarray(create_residual_references(z_ref_train, t_train, reference_ode_parameters))[:,1]
    output_residual_train = output_residual_train.reshape(-1, 1) # We prefer it if the output has a two dimensional shape (n_samples, output_dim) even if the output_dim is 1
    input_residual_train = z_ref_train[:-1]

    output_residual_test = np.asarray(create_residual_references(z_ref_test, t_test, reference_ode_parameters))[:,1]
    output_residual_test = output_residual_test.reshape(-1, 1)
    input_residual_test = z_ref_test[:-1]

    return input_residual_train, output_residual_train, input_residual_test, output_residual_test

# def create_doe_experiments(doe_parameters, method='fullfact'):
#     levels = [len(val) for val in doe_parameters.values()]
#     if method == 'fullfact':
#         doe = fullfact(levels)
#     else:
#         print('Method not supported, using fullfact')
#         doe = fullfact(levels)
#     experiments = []
#     for experiment in doe:
#         experiment_dict = {}
#         for i, key in enumerate(doe_parameters.keys()):
#             experiment_dict[key] = doe_parameters[key][int(experiment[i])]
#         experiments.append(experiment_dict)
#     return tuple(experiments)

# def create_experiment_directory(doe_directory, name):
#     experiment_directory = os.path.join(doe_directory, f'Experiment {name}')
#     if not os.path.exists(experiment_directory):
#         os.mkdir(experiment_directory)
#     return experiment_directory

def create_clean_mini_batch(n_mini_batches, x_ref, t):
    n_timesteps = t.shape[0]
    # Create batches of trajectories
    mini_batch_size = int(n_timesteps/n_mini_batches)
    s = [mini_batch_size * i for i in range(n_mini_batches)]
    x0 = x_ref[s, :]
    targets = [x_ref[s[i]:s[i]+mini_batch_size] for i in range(n_mini_batches)]
    ts = [t[s[i]:s[i]+mini_batch_size] for i in range(n_mini_batches)]
    return x0, targets, ts

def create_hybrid_model(z0, t, ode_parameters, model, layers, mode, restore=False):
    jitted_nn, nn_parameters = create_nn(model, z0, layers)
    hybrid_model = Hybrid_Python(z0=z0,
                                 t=t,
                                 ode_parameters=ode_parameters,
                                 augment_model=jitted_nn,
                                 nn_parameters=nn_parameters,
                                 mode=mode,
                                 restore=restore)
    return hybrid_model

def create_checkpoint_manager(checkpoint_directory, max_to_keep=1, create=True):
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=max_to_keep, create=create)
    checkpoint_manager = orbax.checkpoint.CheckpointManager(checkpoint_directory, orbax_checkpointer, options)
    return checkpoint_manager

def restore_run(directory):
    checkpoint_manager = create_checkpoint_manager(directory, max_to_keep=1)
    step = nn_parameters = checkpoint_manager.latest_step()
    nn_parameters = checkpoint_manager.restore(step)
    return nn_parameters

def save_run(nn_parameters, index, directory, max_to_keep=1):
    checkpoint_manager = create_checkpoint_manager(checkpoint_directory=directory, max_to_keep=max_to_keep)
    save_args = orbax_utils.save_args_from_target(nn_parameters)
    checkpoint_manager.save(index, nn_parameters, save_kwargs={'save_args': save_args})

def run_experiment(dataloader_training, dataloader_testing, model, directory, name, args):

    print(f'Starting Experiment: {name}')

    plot_directory, checkpoint_directory = create_results_subdirectories_(results_directory=directory)
    args['plotting_reference_data']['plot_directory'] = plot_directory
    checkpoint_manager = create_checkpoint_manager(checkpoint_directory)

    start = time.time()
    result = train(model=model,
                   train_dataloader=dataloader_training,
                   test_dataloader=dataloader_testing,
                   device=args['device'],
                   use_multiprocessing=args['use_multiprocessing'],
                   processes=args['processes'],
                   epochs=args['epochs'],
                   particles=args['particles'],
                   particles_batch_size=args['particles_batch_size'],
                   alpha=args['alpha'],
                   sigma=args['sigma'],
                   l=args['l'],
                   dt=args['dt'],
                   anisotropic=args['anisotropic'],
                   eps=args['eps'],
                   partial_update=args['partial_update'],
                   cooling=args['cooling'],
                   checkpoint_interval=args['checkpoint_interval'],
                   problem_type=args['problem_type'],
                   pointers=args['pointers'],
                   residual=args['residual'],
                   plotting_reference_data=args['plotting_reference_data'],
                   checkpoint_manager=checkpoint_manager)

    experiment_time = time.time() - start
    accuracies_train, accuracies_test, losses_train, losses_test, best_index, best_parameters = result
    best_loss_train = float(losses_train[best_index])
    best_loss_test = float(losses_test[best_index])
    print(f'Finished Experiment: {name}')
    print(f'Best Result: Training Loss: {best_loss_train}, Testing Loss: {best_loss_test}, Time: {experiment_time}')
    results_dict = {'Training Loss': best_loss_train, 'Testing Loss': best_loss_test, 'Time': experiment_time}

    # if args.build_plot:
    result = [accuracies_train, accuracies_test, losses_train, losses_test]
    build_plot(args['epochs'], 'Custom', 'VdP', os.path.join(plot_directory, 'Loss.png'), *result)
    model.trajectory_mode()
    pred_train = jnp.array(hybrid_model(input_trajectory_train))
    pred_test = jnp.array(hybrid_model(input_trajectory_test))
    result_plot('Custom', 'VdP', os.path.join(plot_directory, f'Final.png'),
        args['plotting_reference_data']['input_train'], pred_train, args['plotting_reference_data']['input_test'], pred_test,
        np.hstack((t_train, t_test)), np.vstack((args['plotting_reference_data']['z_ref_train'], args['plotting_reference_data']['z_ref_test'])))

    del args['plotting_reference_data']
    experiment_dict = {'setup': args, 'result': results_dict}

    results_file = os.path.join(directory, 'experiment.yaml')
    with open(results_file, 'w') as file:
        yaml.dump(experiment_dict, file)

    save_run(best_parameters, index=0, directory=checkpoint_directory)

    return best_loss_train, best_loss_test, best_parameters, experiment_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--optimisation_method', type=str, default='CBO', help=f'Which type of optimisation method to use: CBO or Adjoint')

    # OLD OPTIONS
    parser.add_argument('--model', type=str, default='CustomMLP', help=f'architecture to use',
                        choices=list(MODELS.keys()))
    parser.add_argument('--dataset', type=str, default='VdP', help='dataset to use',
                        choices=list(DATASETS.keys()))

    # TORCH AND MULTIPROCESSING OPTIONS
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='cuda',
                        help='whether to use GPU (cuda) for accelerated computations or not')
    parser.add_argument('--use_multiprocessing', action='store_true', default=False,
                        help='specify to use multiprocessing for accelerating computations on CPU '
                             '(note, it is impossible to use multiprocessing with GPU)')
    parser.add_argument('--processes', type=int, default=4,
                        help='how many processes to use for multiprocessing')

    # MISC OPTIONS
    parser.add_argument('--doe', type=bool, default=True, help='Whether or not to use the FMU or a python implementation')
    parser.add_argument('--residual', required=False, type=bool, default=True,
                        help='Perform Design of Experiments with NN Parameters and ODE Hyperparameters on Residuals')
    parser.add_argument('--trajectory', required=False, type=bool, default=True,
                        help='Perform Design of Experiments with NN Parameters and ODE Hyperparameters on Trajectory')
    parser.add_argument('--results_directory_name', required=False, type=str, default='Riedl 200 Particles JAX',
                        help='name under which the results should be saved, like plots and such')
    parser.add_argument('--build_plot', required=False, default=True, action='store_true',
                        help='specify to build loss and accuracy plot')
    # parser.add_argument('--plot_path', required=False, type=str, default='demo.png',
    #                     help='path to save the resulting plot')
    parser.add_argument('--checkpoint_interval', type=int, default=100, help='evaluate test accuracy every checkpoint_interval '
                                                                   'samples-level batches')
    parser.add_argument('--fmu', type=bool, default=False, help='Whether or not to use the FMU or a python implementation')

    # GENERAL ODE OPTIONS
    parser.add_argument('--start', type=float, default=0.0, help='Start value of the ODE integration')
    parser.add_argument('--end', type=float, default=10.0, help='End value of the ODE integration')
    parser.add_argument('--n_steps', type=float, default=2001, help='How many integration steps to perform')
    parser.add_argument('--ic', type=list, default=[1.0, 0.0], help='initial_condition of the ODE')
    parser.add_argument('--aug_state', type=bool, default=False, help='Whether or not to use the augemented state for the ODE dynamics')
    parser.add_argument('--aug_dim', type=int, default=4, help='Number of augment dimensions')

    # VdP OPTIONS
    parser.add_argument('--kappa', type=float, default=1.0, help='oscillation constant of the VdP oscillation term')
    parser.add_argument('--mu', type=float, default=8.53, help='damping constant of the VdP damping term')
    parser.add_argument('--mass', type=float, default=1.0, help='mass constant of the VdP system')
    parser.add_argument('--stimulate', type=bool, default=False, help='Whether or not to use the stimulated dynamics')

    # parser.add_argument('--simple_problem', type=bool, default=False, help='Whether or not to use a simple damped oscillator instead of VdP')

    # NEURAL NETWORK OPTIONS
    parser.add_argument('--layers', type=int, default=2, help='Number of hidden layers')
    parser.add_argument('--layer_size', type=int, default=40, help='Number of neurons in a hidden layer')

    # GENERAL OPTIMIZER OPTIONS
    # CBO OPTIONS
    parser.add_argument('--epochs', type=int, default=100, help='train for EPOCHS epochs')
    parser.add_argument('--batch_size', type=int, default=100, help='batch size (for samples-level batching)')
    parser.add_argument('--particles', type=int, default=100, help='')
    parser.add_argument('--particles_batch_size', type=int, default=10, help='batch size '
                                                                             '(for particles-level batching)')
    parser.add_argument('--alpha', type=float, default=100.0, help='alpha from CBO dynamics')
    parser.add_argument('--sigma', type=float, default=0.1 ** 0.5, help='sigma from CBO dynamics')
    parser.add_argument('--l', type=float, default=0.1, help='lambda from CBO dynamics')
    parser.add_argument('--dt', type=float, default=1.0, help='dt from CBO dynamics')
    parser.add_argument('--anisotropic', type=bool, default=True, help='whether to use anisotropic or not')
    parser.add_argument('--eps', type=float, default=1e-4, help='threshold for additional random shift')
    parser.add_argument('--partial_update', type=bool, default=False, help='whether to use partial or full update')
    parser.add_argument('--cooling', type=bool, default=False, help='whether to apply cooling strategy')

    parser.add_argument('--restore', required=False, type=bool, default=False,
                        help='restore previous parameters')
    parser.add_argument('--restore_name', required=False, type=str, default='plot_test_75',
                        help='directory from which results will be restored')
    parser.add_argument('--overwrite', required=False, type=bool, default=True,
                        help='overwrite previous result')

    parser.add_argument('--problem_type', required=False, type=str, default='regression',
                            help='problem type')
    args = parser.parse_args()

    if args.batch_size == -1:
        args.batch_size = args.n_steps
        if args.residual:
            # For residuals we have one sample less available because of the finite difference calculation
            args.batch_size -= 1

    path = os.path.abspath(__file__)
    directory = os.path.sep.join(path.split(os.path.sep)[:-1])
    file_path = get_file_path(path)
    results_directory = create_results_directory(directory=directory, results_directory_name=args.results_directory_name)

    # CREATE REFERENCE SOLUTION
    ####################################################################################
    z0 = np.array(args.ic)
    reference_ode_parameters = [args.kappa, args.mu, args.mass]
    t_train, z_ref_train, t_test, z_ref_test, z0_test = create_reference_solution(args.start, args.end, args.n_steps, z0, reference_ode_parameters, ode_integrator=f_euler_python)

    # CONVERT THE REFERENCE DATA TO A DATASET
    ####################################################################################
    if args.residual:
        input_residual_train, output_residual_train, input_residual_test, output_residual_test = create_residual_reference_solution(t_train, z_ref_train, t_test, z_ref_test, reference_ode_parameters)
        dataset_residual_train = create_generic_dataset(np.array(input_residual_train), np.array(output_residual_train))
        dataset_residual_test = create_generic_dataset(np.array(input_residual_test), np.array(output_residual_test))
        dataloader_residual_train, dataloader_residual_test = load_generic_dataloaders(train_dataset=dataset_residual_train,
                                                                     train_batch_size=args.batch_size if args.batch_size < len(dataset_residual_train) else len(dataset_residual_train),
                                                                     test_dataset=dataset_residual_test,
                                                                     test_batch_size=args.batch_size if args.batch_size < len(dataset_residual_test) else len(dataset_residual_test))

    # Each particle contains the time and reference data as input we just
    input_trajectory_train = np.hstack((t_train.reshape(-1,1), z_ref_train))
    input_trajectory_test = np.hstack((t_test.reshape(-1,1), z_ref_test))
    dataset_trajectory_train = create_generic_dataset(input_trajectory_train, np.array(z_ref_train))
    dataset_trajectory_test = create_generic_dataset(input_trajectory_test, np.array(z_ref_test))
    dataloader_trajectory_train, dataloader_trajectory_test = load_generic_dataloaders(train_dataset=dataset_trajectory_train,
                                                                    train_batch_size=args.batch_size if args.batch_size < len(dataset_trajectory_train) else len(dataset_trajectory_train),
                                                                    test_dataset=dataset_trajectory_test,
                                                                    test_batch_size=args.batch_size if args.batch_size < len(dataset_trajectory_test) else len(dataset_trajectory_test))

    # Save the reference data for plotting during training
    plotting_reference_data = {'input_train': t_train,
                               'input_test': t_test,
                               'input_trajectory_train': input_trajectory_train,
                               'input_trajectory_test': input_trajectory_test,
                               'z_ref_train': z_ref_train,
                               'z_ref_test': z_ref_test,
                               'plot_directory': results_directory}

    if args.optimisation_method == 'CBO':
        print('Optimising with Consensus Based Optimisation (CBO)')
        # TORCH DEVICES
        ####################################################################################
        # device = args.device
        # if args.device == 'cuda' and not torch.cuda.is_available():
        #     print('Cuda is unavailable. Using CPU instead.')
        #     device = 'cpu'
        use_multiprocessing = args.use_multiprocessing
        # if device != 'cpu' and use_multiprocessing:
        #     print('Unable to use multiprocessing on GPU')
        #     use_multiprocessing = False
        # device = torch.device(device)

        results_name =  'results.txt'
        setup_file_name = 'best_result.yaml'
        parameter_file_name = 'parameters.pt'

        results_file = os.path.join(results_directory, results_name)
        setup_file = os.path.join(results_directory, setup_file_name)

        experiment_args = {'device': args.device,
                           'use_multiprocessing': args.use_multiprocessing,
                           'processes': args.processes,
                           'epochs': args.epochs,
                           'particles': args.particles,
                           'particles_batch_size': args.particles_batch_size,
                           'alpha': args.alpha,
                           'sigma': args.sigma,
                           'l': args.l,
                           'dt': args.dt,
                           'anisotropic': args.anisotropic,
                           'eps': args.eps,
                           'partial_update': args.partial_update,
                           'cooling': args.cooling,
                           'checkpoint_interval': args.checkpoint_interval,
                           'problem_type': args.problem_type,
                           'pointers': None,
                           'residual': True,
                           'plotting_reference_data': plotting_reference_data}
        if args.restore:
            restore_directory = os.path.join(directory, args.restore_name)
            if not os.path.exists(restore_directory):
                print('Restore path does not exist')
                exit(1)
            restore_results_file_path = os.path.join(restore_directory, results_name)
            restore_setup_file = os.path.join(restore_directory, setup_file_name)

            # Try to restore a previous experiment run
            with open(restore_setup_file, 'r') as file:
                restore_best_experiment = yaml.safe_load(file)

            for k,v in restore_best_experiment.items():
                vars(args)[k] = v

        if args.doe:
            print('Performing Design of Experiment (DoE)')
            # DoE Parameters

            if args.residual:
                # residual_doe_parameters = OrderedDict({'alpha': [1e0, 1e2]})

                residual_doe_parameters = OrderedDict({'alpha': [1, 100],
                                                       'sigma': [0.1**0.5, 0.4**0.5],
                                                       'l': [1.0, 0.1],
                                                       'dt': [0.1, 0.01],
                                                       'particles':[200],
                                                       'cooling': [False],
                                                       'layer_size': [40],
                                                       'layers': [2]})
                print(f'Residual DoE Parameters: {residual_doe_parameters}')

            if args.trajectory:
                if args.residual:
                    trajectory_doe_parameters = OrderedDict({'batch_size': [100, 200]})
                else:
                    trajectory_doe_parameters = OrderedDict({'alpha': [1e0, 1e2],
                                                            'sigma': [0.1**0.5, 0.4**0.5],
                                                            'l': [1.0, 0.1, 0.01],
                                                            'dt': [1.0, 0.1, 0.01],
                                                            'particles':[10, 100],
                                                            'batch_size': [100, 200],
                                                            'cooling': [True, False],
                                                            'layer_size': [20, 40],
                                                            'n_layers': [1, 2, 3]})
                print(f'Trajectory DoE Parameters: {trajectory_doe_parameters}')

            experiment_losses = []
            experiment_strings = []
            best_experiment = {'n_exp': None,
                               'setup': {},
                               'loss': np.inf,
                               'loss_test': np.inf,
                               'residual_loss': np.inf,
                               'residual_loss_test': np.inf,
                               'model': None,
                               'time': 0.0}

            if args.restore:
                # If we restore the best setup of a previous DoE run the we omit all the
                # parameters checked in the previous run
                print('Restoring Parameters from previous DoE run. Removing already checked parameters...')
                for k,v in restore_best_experiment['setup'].items():
                    try:
                        residual_doe_parameters[k] = v
                    except:
                        pass
                    try:
                        trajectory_doe_parameters[k] = v
                    except:
                        pass

            if args.fmu:
                print('DoE for FMU not yet implemented')
                raise NotImplementedError
            else:
                if args.residual:
                    print('Starting Residual DoE Run...')
                    if len(residual_doe_parameters) != 0:
                        doe_start_time = time.time()

                        with open(results_file, 'a') as file:
                            file.writelines(f'VdP Setup: kappa: {args.kappa}, mu: {args.mu}, mass: {args.mass}, Start: {args.start}, End: {args.end}, Steps: {args.n_steps}')
                            file.write('\n')

                        experiments = create_doe_experiments(residual_doe_parameters, method='fullfact')

                        for n_exp, experiment in enumerate(experiments):
                            experiment_name = f'DoE Residual {n_exp}'
                            experiment_directory = create_experiment_directory(results_directory, experiment_name)

                            # Write the experiment values into the args NameSpace, from which parameter values get drawn
                            for k, v in experiment.items():
                                experiment_args[k] = v

                            layers = layers = [2] + [args.layer_size]*args.layers + [1]
                            hybrid_model = create_hybrid_model(z0, t_train, reference_ode_parameters, args.model, layers, mode='residual')
                            experiment_args['plotting_reference_data'] = plotting_reference_data

                            start = time.time()
                            result = run_experiment(dataloader_training=dataloader_residual_train,
                                                    dataloader_testing=dataloader_residual_test,
                                                    model=hybrid_model,
                                                    directory = experiment_directory,
                                                    name=experiment_name,
                                                    args=experiment_args)
                            experiment_time = time.time() - start
                            training_loss, testing_loss, best_parameters, full_experiment_dict = result

                            full_experiment_dict['n_exp'] = n_exp
                            with open(results_file, 'a') as file:
                                file.write(f'Experiment {n_exp}: Best training loss: {training_loss}, Best testing loss: {testing_loss}, Time: {experiment_time}')
                                file.write('\n')

                            if testing_loss < best_experiment['residual_loss_test']:
                                best_experiment['n_exp'] = n_exp
                                best_experiment['setup'] = experiment
                                best_experiment['residual_loss'] = training_loss
                                best_experiment['residual_loss_test'] = testing_loss
                                best_experiment['model'] = hybrid_model
                                best_experiment['time'] = experiment_time
                                save_run(best_parameters, 0, results_directory)
                                with open(setup_file, 'w') as file:
                                    yaml.dump(full_experiment_dict, file)

                        with open(results_file, 'a') as file:
                            best_exp = best_experiment['n_exp']
                            file.write(f'Best experiment: {best_exp}')
                            file.write('\n')

                    else:
                        # NO DoE, just one regular run on of CBO on Residuals
                        experiment_name = 'Single Residual Doe Run'

                        layers = layers = [2] + [args.layer_size]*args.layers + [1]
                        hybrid_model = create_hybrid_model(z0, t_train, reference_ode_parameters, args.model, layers, mode='residual')
                        experiment_args['plotting_reference_data'] = plotting_reference_data

                        start = time.time()
                        result = run_experiment(dataloader_training=dataloader_residual_train,
                                                dataloader_testing=dataloader_residual_test,
                                                model=hybrid_model,
                                                directory = results_directory,
                                                name=experiment_name,
                                                args=experiment_args)
                        best_training_loss, best_testing_loss, best_parameters = result

                if args.trajectory:
                    if len(trajectory_doe_parameters) != 0:
                        doe_start_time = time.time()

                        with open(results_file, 'a') as file:
                            file.writelines(f'VdP Setup: kappa: {args.kappa}, mu: {args.mu}, mass: {args.mass}, Start: {args.start}, End: {args.end}, Steps: {args.n_steps}')
                            file.write('\n')

                        experiments = create_doe_experiments(trajectory_doe_parameters, method='fullfact')

                        if args.residual:
                            # We did a residual DoE run before. Load the best result
                            with open(setup_file, 'r') as file:
                                full_experiment_dict = yaml.safe_load(file)

                            for k,v in full_experiment_dict['setup'].items():
                                experiment_args[k] = v

                        # EXECUTE DoE
                        for n_exp, experiment in enumerate(experiments):
                            experiment_name = f'Trajectory {n_exp}'
                            experiment_directory = create_experiment_directory(results_directory, experiment_name)

                            # Write the experiment values into the args NameSpace
                            for k, v in experiment.items():
                                experiment_args[k] = v

                            if args.residual:
                                nn_parameters = restore_run(results_directory)
                                restore = True
                            else:
                                restore = False

                            layers = layers = [2] + [args.layer_size]*args.layers + [1]
                            hybrid_model = create_hybrid_model(z0, t_train, reference_ode_parameters, args.model, layers, mode='trajectory', restore=restore)
                            if restore:
                                hybrid_model.nn_parameters = nn_parameters
                            experiment_args['plotting_reference_data'] = plotting_reference_data

                            start = time.time()
                            result = run_experiment(dataloader_training=dataloader_residual_train,
                                                    dataloader_testing=dataloader_residual_test,
                                                    model=hybrid_model,
                                                    directory=experiment_directory,
                                                    name=experiment_name,
                                                    args=experiment_args)
                            experiment_time = time.time() - start

                            training_loss, testing_loss, best_parameters, full_experiment_dict = result
                            full_experiment_dict['n_exp'] = n_exp

                            with open(results_file, 'a') as file:
                                file.write(f'Experiment {n_exp}: Best training loss: {training_loss}, Best testing loss: {testing_loss}, Time: {experiment_time}')
                                file.write('\n')

                            if testing_loss < best_experiment['residual_loss_test']:
                                best_experiment['n_exp'] = n_exp
                                best_experiment['setup'] = experiment
                                best_experiment['residual_loss'] = training_loss
                                best_experiment['residual_loss_test'] = testing_loss
                                best_experiment['model'] = hybrid_model
                                best_experiment['time'] = experiment_time
                                save_run(best_parameters, 0, results_directory)
                                with open(setup_file, 'w') as file:
                                    yaml.dump(full_experiment_dict, file)

                        with open(results_file, 'a') as file:
                            best_exp = best_experiment['n_exp']
                            file.write(f'Best experiment: {best_exp}')
                            file.write('\n')

                    else:
                        f_euler = f_euler_python
                        experiment_name = f'Trajectory {n_exp}'

                        layers = layers = [2] + [args.layer_size]*args.layers + [1]
                        hybrid_model = create_hybrid_model(z0, t_train, reference_ode_parameters, args.model, layers, mode='trajectory')
                        experiment_args['plotting_reference_data'] = plotting_reference_data

                        start = time.time()
                        result = run_experiment(dataloader_training=dataloader_residual_train,
                                                dataloader_testing=dataloader_residual_test,
                                                model=hybrid_model,
                                                directory = results_directory,
                                                name=experiment_name,
                                                args=experiment_args)
                        best_training_loss, best_testing_loss, best_parameters, full_experiment_dict = result

                        save_run(best_parameters, 0, results_directory)
                        with open(setup_file, 'w') as file:
                            yaml.dump(full_experiment_dict, file)
        else:

            if args.fmu:
                if args.trajectory:
                        # ODE SETUP
                        ####################################################################################
                        # Training Setup
                        t_train = np.linspace(args.start, args.end, args.n_steps)

                        # Test Setup
                        t_start_test = args.end
                        t_end_test = args.end + (args.end - args.start)*0.5
                        n_steps_test = int(args.n_steps* 0.5)
                        t_test = np.linspace(t_start_test, t_end_test, n_steps_test)

                        reference_ode_parameters = [args.kappa, args.mu, args.mass]

                        # FMU SETUP
                        ####################################################################################
                        fmu_filename = 'Van_der_Pol_damping_input.fmu'
                        path = os.path.abspath(__file__)
                        fmu_filename = '/'.join(path.split('/')[:-1]) + '/' + fmu_filename
                        fmu_evaluator = FMUEvaluator(fmu_filename, args.start, args.end)
                        pointers = fmu_evaluator.get_pointers()

                        z_ref_train = f_euler_fmu(z0=args.ic, t=t_train, fmu_evaluator=fmu_evaluator, model=damping, model_parameters=args.mu, pointers=pointers)
                        fmu_evaluator.reset_fmu(t_start_test, t_end_test)
                        z_ref_test = f_euler_fmu(z0=z_ref_train[-1], t=t_test, fmu_evaluator=fmu_evaluator, model=damping, model_parameters=args.mu, pointers=pointers)
                        fmu_evaluator.reset_fmu(args.start, args.end)


                        # CONVERT THE REFERENCE DATA TO A DATASET
                        ####################################################################################

                        train_dataset = create_generic_dataset(np.array(t_train), np.array(z_ref_train))
                        test_dataset = create_generic_dataset(np.array(t_test), np.array(z_ref_test))

                        train_dataloader, test_dataloader = load_generic_dataloaders(train_dataset=train_dataset,
                                                                                    train_batch_size=args.n_steps,
                                                                                    test_dataset=test_dataset,
                                                                                    test_batch_size=n_steps_test)

                        # TRAINING
                        ####################################################################################
                        layers = layers = [2] + [args.layer_size]*args.layers + [1]
                        hybrid_model = create_hybrid_model(z0, t_train, reference_ode_parameters, args.model, layers, mode='trajectory')

                        start_time = time.time()
                        result = train(model=hybrid_model,
                                    train_dataloader=train_dataloader,
                                    test_dataloader=test_dataloader,
                                    device=None,
                                    use_multiprocessing=use_multiprocessing,
                                    processes=args.processes,
                                    epochs=args.epochs,
                                    particles=args.particles,
                                    particles_batch_size=args.particles_batch_size,
                                    alpha=args.alpha,
                                    sigma=args.sigma,
                                    l=args.l,
                                    dt=args.dt,
                                    anisotropic=args.anisotropic,
                                    eps=args.eps,
                                    partial_update=args.partial_update,
                                    cooling=args.cooling,
                                    checkpoint_interval=args.checkpoint_interval,
                                    problem_type='regression',
                                    pointers=pointers,
                                    plotting_reference_data=plotting_reference_data)
                        print(f'Elapsed time: {time.time() - start_time} seconds')
                        if args.build_plot:
                            build_plot(args.epochs, args.model, args.dataset, os.path.join(results_directory, f'Loss.png'),
                                    *result)

                            hybrid_model.t = t_train
                            hybrid_model.z0 = z_ref_train[0]
                            pred_train = jnp.array(hybrid_model(pointers))

                            hybrid_model.t = t_test
                            hybrid_model.z0 = z_ref_test[0]
                            pred_test = jnp.array(hybrid_model(pointers))

                            result_plot('Custom', 'VdP', os.path.join(results_directory, f'Final.png'),
                                        t_train, pred_train, t_test, pred_test,
                                        np.hstack((t_train, t_test)), np.vstack((z_ref_train, z_ref_test)))
            else:
                if args.residual:
                    print('Single Residual Run...')
                    experiment_name = f'Single Residual Run'

                    layers = layers = [2] + [args.layer_size]*args.layers + [1]
                    hybrid_model = create_hybrid_model(z0, t_train, reference_ode_parameters, args.model, layers, mode='trajectory')
                    experiment_args['plotting_reference_data'] = plotting_reference_data

                    start = time.time()
                    result = run_experiment(dataloader_training=dataloader_residual_train,
                                            dataloader_testing=dataloader_residual_test,
                                            model=hybrid_model,
                                            directory = results_directory,
                                            name=experiment_name,
                                            args=experiment_args)
                    best_training_loss, best_testing_loss, best_parameters, full_experiment_dict = result

                    save_run(best_parameters, 0, results_directory)
                    with open(setup_file, 'w') as file:
                        yaml.dump(full_experiment_dict, file)

                if args.trajectory:
                    print('Single Trajectory Run...')
                    experiment_name = f'Single Trajectory Run'
                    f_euler = f_euler_python

                    layers = layers = [2] + [args.layer_size]*args.layers + [1]
                    hybrid_model = create_hybrid_model(z0, t_train, reference_ode_parameters, args.model, layers, mode='trajectory')
                    experiment_args['plotting_reference_data'] = plotting_reference_data

                    start = time.time()
                    result = run_experiment(dataloader_training=dataloader_trajectory_train,
                                            dataloader_testing=dataloader_trajectory_test,
                                            model=hybrid_model,
                                            directory = results_directory,
                                            name=experiment_name,
                                            args=experiment_args)
                    best_training_loss, best_testing_loss, best_parameters, full_experiment_dict = result

                    save_run(best_parameters, 0, results_directory)
                    with open(setup_file, 'w') as file:
                        yaml.dump(full_experiment_dict, file)
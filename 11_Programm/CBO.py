import os
import sys
import shutil
import torch
import torch.nn.functional as F
import numpy as np
import argparse
import time
import yaml
import matplotlib.pyplot as plt
import copy

import jax
from jax import lax, jit
import jax.numpy as jnp
# this only works on startup!
from jax.config import config
config.update("jax_enable_x64", True)
from functools import partial
import orbax.checkpoint

sys.path.insert(1, os.getcwd())
# from plot_results import plot_results, plot_losses, get_file_path
# from plot_results import plot_losses, get_file_path, visualise_wb
from fmu_helper import FMUEvaluator
from cbo_in_python.src.torch_.models import *
# from cbo_in_python.src.datasets import load_mnist_dataloaders, load_parabola_dataloaders, f
from cbo_in_python.src.datasets import create_generic_dataset, load_generic_dataloaders
from cbo_in_python.src.torch_.optimizer import Optimizer
from cbo_in_python.src.torch_.loss import Loss
from torch.utils.data import Dataset, DataLoader

from utils import build_plot, result_plot_multi_dim, create_results_directory, create_results_subdirectories, create_doe_experiments, create_experiment_directory, visualise_wb

import logging
from sklearn.preprocessing import MinMaxScaler
from typing import List

MODELS = {
    'SimpleMLP': SimpleMLP,
    'TinyMLP': TinyMLP,
    'SmallMLP': SmallMLP,
    'LeNet1': LeNet1,
    'LeNet5': LeNet5,
    'CustomMLP': CustomMLP
}

DATASETS = {
    'VdP': ''
}

# # PYTHON ONLY ODEs
# @jit
# def ode(x, t, variables):
#     '''Calculates the right hand side of the original ODE.'''
#     kappa = variables[0]
#     mu = variables[1]
#     mass = variables[2]
#     derivative = jnp.array([x[1],
#                            -kappa*x[0]/mass + (mu*(1-x[0]**2))/mass])
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
# def ode_stim(x, t, variables):
#     '''Calculates the right hand side of the original ODE.'''
#     kappa = variables[0]
#     mu = variables[1]
#     mass = variables[2]
#     derivative = jnp.array([x[1],
#                            -kappa*x[0]/mass + (mu*(1-x[0]**2)*x[1])/mass + 1.2*jnp.cos(jnp.pi/5*t)])
#     return derivative

# @jit
# def ode_stim_res(x, t, variables):
#     '''Calculates the right hand side of the original ODE.'''
#     kappa = variables[0]
#     mu = variables[1]
#     mass = variables[2]
#     derivative = jnp.array([x[1],
#                            -kappa*x[0]/mass + 1.2*jnp.cos(jnp.pi/5*t)])
#     return derivative

def create_residual_references(x_ref, t, variables):
    x_dot = (x_ref[1:] - x_ref[:-1])/(t[1:] - t[:-1]).reshape(-1,1)
    v_ode = jax.vmap(lambda x_ref, t, variables: ode_res(x_ref, t, variables), in_axes=(0, 0, None))
    residual = x_dot - v_ode(x_ref[:-1], t[:-1], variables)
    return residual


def J_residual(inputs, outputs, neural_network, parameters):
    def squared_error(input, output, neural_network, parameters):
        pred = neural_network(parameters, input, neural_network)
        return (output-pred)**2
    error_function = partial(squared_error, neural_network=neural_network, parameters=parameters)
    return jnp.mean(jax.vmap(error_function)(inputs, outputs), axis=0)[0]


# Managing class for Hybrid model with FMU
class Hybrid_FMU(nn.Module):
    def __init__(self, fmu_model, augment_model, x0, t, mode='trajectory'):
        super(Hybrid_FMU, self).__init__()

        self.fmu_model = fmu_model
        self.augment_model = augment_model
        self.x0 = x0
        self.t = t
        self.pointers = fmu_model.pointers
        self.mode = mode

        if mode=='residual':
            self.forward = self.forward_residual
        elif mode=='trajectory':
            self.forward = self.forward_trajectory
        else:
            print('Mode not recognised. Using residual mode')
            self.forward = self.forward_residual

    def residual_mode(self):
        self.mode = 'residual'
        self.forward = self.forward_residual

    def trajectory_mode(self):
        self.mode = 'trajectory'
        self.forward = self.forward_trajectory

    def set_trajectory_variables(self, x0, t):
        self.x0 = x0
        self.t = t

    def augment_model_function(self, parameters, inputs):
        # The augment_model is currently a pytorch model, which just takes
        # the input. It has its own parameters saved internally.
        # The f_euler function expects a model which needs its paramaters
        # given when it is called: y = augment_model_function(parameters, input)
        # f_euler provides the input to the augment_model as numpy array
        # but we can only except tensors, so convert
        return self.augment_model(torch.tensor(inputs)).detach().numpy()

    def forward_trajectory(self):
        '''Applies euler to the VdP ODE by calling the fmu; returns the trajectory'''
        t = self.t
        x0 = self.x0
        x = np.zeros((t.shape[0], x0.shape[0]))
        x[0] = x0
        # Forward the initial state to the FMU
        self.fmu_model.setup_initial_state(x0)
        times = []
        for i in range(len(t)-1):
            # start = time.time()
            status = self.fmu_model.fmu.setTime(t[i])
            dt = t[i+1] - t[i]
            enterEventMode, terminateSimulation, _, _ = self.fmu_model.evaluate_fmu(t[i], dt, self.augment_model_function, None)
            x[i+1] = x[i] + dt * self.fmu_model.pointers.dx
            if terminateSimulation:
                break
        return x

    def forward_residual(self, inputs):
        return self.augment_model(inputs)


# Python only Hybrid model
class Hybrid_Python(nn.Module):
    def __init__(self, variables, augment_model, x0, t, hybrid_ode, mode='trajectory'):
        super(Hybrid_Python, self).__init__()

        self.variables = variables
        self.augment_model = augment_model
        self.x0 = x0
        self.t = t
        self.hybrid_ode = hybrid_ode
        self.mode = mode

        if mode=='residual':
            self.forward = self.forward_residual
        elif mode=='trajectory':
            self.forward = self.forward_trajectory
        else:
            print('Mode not recognised. Using residual mode')
            self.forward = self.forward_residual

    def residual_mode(self):
        self.mode = 'residual'
        self.forward = self.forward_residual

    def trajectory_mode(self):
        self.mode = 'trajectory'
        self.forward = self.forward_trajectory

    def set_trajectory_variables(self, x0, t):
        self.x0 = x0
        self.t = t

    def augment_model_function(self, parameters, inputs):
        # The augment_model is currently a pytorch model, which just takes
        # the input. It has its own parameters saved internally.
        # The f_euler function expects a model which needs its paramaters
        # given when it is called: y = augment_model_function(parameters, input)
        # f_euler provides the input to the augment_model as numpy array
        # but we can only except tensors, so convert
        return self.augment_model(torch.tensor(inputs)).detach().numpy()

    def forward_residual(self, inputs):
        return self.augment_model(inputs)

    def forward_trajectory(self, stim=False):
        # t = self.t
        # x0 = self.x0
        # x = np.zeros((t.shape[0], x0.shape[0]))
        # x[0] = x0
        # Forward the initial state to the FMU
        times = []
        if stim:
            self.hybrid_ode = self.hybrid_ode_stim
        else:
            # NN parameters are saved inside the augment_model and therefore not needed as input
            self.hybrid_ode = self.hybrid_ode
        x = self.hybrid_euler(variables=self.variables, parameters=None)
        return x

    def hybrid_euler(self, variables, parameters):
        '''Applies forward Euler to the hybrid ODE and returns the trajectory'''
        t = self.t
        x0 = self.x0
        x = np.zeros((t.shape[0], x0.shape[0]))
        # x = x.at[0].set(x0)
        x[0] = x0
        i = np.asarray(range(t.shape[0]))
        # We can replace the loop over the time by a lax.scan this is 3 times as fast: e.g.: 0.32-0.26 -> 0.11-0.9
        # euler_body_func = partial(self.hybrid_step, t=t, variables=variables, parameters=parameters)
        # final, result = lax.scan(euler_body_func, x0, i)
        # x = x.at[1:].set(result[:-1])
        for i in range(len(t)-1):
            dt = t[i+1] - t[i]
            x[i+1] = x[i] + dt * self.hybrid_ode(x[i], t[i], variables, parameters, self.augment_model_function)
        return x

    # def hybrid_ode(self, x, t, variables, parameters):
    #     '''Calculates the right hand side of the hybrid ODE, where
    #     the damping term is replaced by the neural network'''
    #     kappa = variables[0]
    #     mu = variables[1]
    #     mass = variables[2]
    #     derivative = np.array([np.array((x[1],)),
    #                            np.array((-kappa*x[0]/mass,)) + self.augment_model_function(parameters, x)]).flatten()
    #     return derivative

    # def hybrid_ode_stim(self, x, t, variables, parameters):
    #     '''Calculates the right hand side of the hybrid ODE, where
    #     the damping term is replaced by the neural network'''
    #     kappa = variables[0]
    #     mu = variables[1]
    #     mass = variables[2]
    #     derivative = np.array([np.array((x[1],)),
    #                             np.array((-kappa*x[0]/mass,)) + self.augment_model_function(parameters, x) + np.array(1.2*np.cos(np.pi/5*t))]).flatten()
    #     return derivative

def f_euler_fmu(x0, t, fmu_evaluator: FMUEvaluator, model, model_parameters, make_zero_input_prediction=False):
    '''Applies euler to the VdP ODE by calling the fmu; returns the trajectory'''
    x = np.zeros((t.shape[0], x0.shape[0]))
    x[0] = x0
    # Forward the initial state to the FMU
    fmu_evaluator.setup_initial_state(x0)
    times = []
    if fmu_evaluator.training:
        dfmu_dx_trajectory = []
        dfmu_dinput_trajectory = []
    derivatives_list = []
    for i in range(len(t)-1):
        status = fmu_evaluator.fmu.setTime(t[i])
        dt = t[i+1] - t[i]
        if fmu_evaluator.training:
            enterEventMode, terminateSimulation, dfmu_dx_at_t, dfmu_dinput_at_t = fmu_evaluator.evaluate_fmu(t[i], dt, model, model_parameters)
            x[i+1] = x[i] + dt * fmu_evaluator.pointers.dx
            if save_derivatives:
                derivatives_list.append(copy.copy(fmu_evaluator.pointers.dx))
            dfmu_dx_trajectory.append(dfmu_dx_at_t)
            dfmu_dinput_trajectory.append(dfmu_dinput_at_t)
        else:
            enterEventMode, terminateSimulation, zero_input_prediction, _ = fmu_evaluator.evaluate_fmu(t[i], dt, model, model_parameters, make_zero_input_prediction)
            x[i+1] = x[i] + dt * fmu_evaluator.pointers.dx
            if make_zero_input_prediction:
                derivatives_list.append(zero_input_prediction)
        if terminateSimulation:
            break

    # We get on jacobian less then we get datapoints, since we get the jacobian
    # with every derivative we calculate, and we have one datapoint already given
    # at the start
    if fmu_evaluator.training:
        enterEventMode, terminateSimulation, dfmu_dx_at_t, dfmu_dinput_at_t = fmu_evaluator.evaluate_fmu(t[-1], t[-1]-t[-2], model, model_parameters)
        dfmu_dx_trajectory.append(dfmu_dx_at_t)
        dfmu_dinput_trajectory.append(dfmu_dinput_at_t)
        dfmu_dinput_trajectory = jnp.asarray(dfmu_dinput_trajectory)
        while len(dfmu_dinput_trajectory.shape) <= 2:
            dfmu_dinput_trajectory = jnp.expand_dims(dfmu_dinput_trajectory, -1)
        return x, np.asarray(dfmu_dx_trajectory), jnp.asarray(dfmu_dinput_trajectory)
    else:
        if make_zero_input_prediction:
            return x, np.asarray(derivatives_list)
        else:
            return x

def euler(x0, t, variables):
    '''Applies forward Euler to the original ODE and returns the trajectory'''
    x = jnp.zeros((t.shape[0], x0.shape[0]))
    x = x.at[0].set(x0)
    i = jnp.asarray(range(t.shape[0]))
    euler_body_func = partial(step, t=t, variables = variables)
    final, result = lax.scan(euler_body_func, x0, i)
    x = x.at[1:].set(result[:-1])
    return x

def step(prev_x, i, t, variables):
    t = jnp.asarray(t)
    dt = t[i+1] - t[i]
    next_x = prev_x + dt * ode(prev_x, t[i], variables)
    return next_x, next_x

def _evaluate_reg(outputs, y_, loss_fn):
    with torch.no_grad():
        loss = loss_fn(outputs, y_)
    return loss

def create_reference_solution(start, end, n_steps, x0, reference_variables, ode_integrator):

    t_train = np.linspace(start, end, n_steps)
    x_ref_train = np.asarray(ode_integrator(x0, t_train, reference_variables))

    # Generate the reference data for testing
    x0_test = x_ref_train[-1]
    n_steps_test = int(n_steps * 0.5)
    t_test = np.linspace(end, (end - start) * 1.5, n_steps_test)
    x_ref_test = np.asarray(ode_integrator(x0_test, t_test, reference_variables))

    return t_train, x_ref_train, t_test, x_ref_test, x0_test

def create_residual_reference_solution(t_train, x_ref_train, t_test, x_ref_test, reference_variables):

    # CREATE RESIDUALS FROM TRAJECTORIES
    train_residual_outputs = np.asarray(create_residual_references(x_ref_train, t_train, reference_variables))[:,1]
    train_residual_outputs = train_residual_outputs.reshape(-1, 1) # We prefer it if the output has a two dimensional shape (n_samples, output_dim) even if the output_dim is 1
    train_residual_inputs = x_ref_train[:-1]

    test_residual_outputs = np.asarray(create_residual_references(x_ref_test, t_test, reference_variables))[:,1]
    test_residual_outputs = test_residual_outputs.reshape(-1, 1)
    test_residual_inputs = x_ref_test[:-1]

    return train_residual_inputs, train_residual_outputs, test_residual_inputs, test_residual_outputs

def create_clean_mini_batch(n_mini_batches, x_ref, t):
    n_timesteps = t.shape[0]
    # Create batches of trajectories
    mini_batch_size = int(n_timesteps/n_mini_batches)
    s = [mini_batch_size * i for i in range(n_mini_batches)]
    x0 = x_ref[s, :]
    targets = [x_ref[s[i]:s[i]+mini_batch_size] for i in range(n_mini_batches)]
    ts = [t[s[i]:s[i]+mini_batch_size] for i in range(n_mini_batches)]
    return x0, targets, ts

def restore_parameters(hybrid_model, parameter_file):
    try:
        ckpt = torch.load(parameter_file)
        hybrid_model.load_state_dict(ckpt['model_state_dict'])
        print('Continuing training with previous parameters')
    except:
        print('Could not find model to load')
    return hybrid_model

def plot_loss_and_prediction(accuracies_train,
                             accuracies_test,
                             losses_train,
                             losses_test,
                             t_train,
                             t_test,
                             x_ref_train,
                             x_ref_test,
                             pred_train,
                             pred_test,
                             epochs,
                             model_name,
                             dataset_name,
                             reference_variables,
                             results_directory):
    t_train = plotting_reference_data['t_train']
    t_test = plotting_reference_data['t_test']
    x_ref_train = plotting_reference_data['x_ref_train']
    x_ref_test = plotting_reference_data['x_ref_test']
    results_directory = plotting_reference_data['results_directory']

    build_plot(args.epochs, args.model, args.dataset, os.path.join(results_directory, 'Loss.png'), *result)

    # If we decide to plot the results, we want to plot on the trajectory
    # and not only on the residuals we train on in this case.
    # We therefore need a Non residual model
    augment_model = hybrid_model.augment_model
    plot_model = Hybrid_Python(reference_variables, augment_model, x_ref_train[0], t_train, mode='trajectory')
    pred_train = torch.tensor(plot_model())

    plot_model.t = t_test
    plot_model.x0 = x0_test
    pred_test = torch.tensor(plot_model())

    result_plot_multi_dim('Custom', 'VdP', os.path.join(results_directory, f'Final.png'),
                t_train, pred_train, t_test, pred_test,
                np.hstack((t_train, t_test)), np.vstack((x_ref_train, x_ref_test)))


# def plot_loss_and_prediction(result, hybrid_model, args, reference_variables, plotting_reference_data):
#     t_train = plotting_reference_data['t_train']
#     t_test = plotting_reference_data['t_test']
#     x_ref_train = plotting_reference_data['x_ref_train']
#     x_ref_test = plotting_reference_data['x_ref_test']
#     results_directory = plotting_reference_data['results_directory']

#     build_plot(args.epochs, args.model, args.dataset, os.path.join(results_directory, 'Loss.png'), *result)

#     # If we decide to plot the results, we want to plot on the trajectory
#     # and not only on the residuals we train on in this case.
#     # We therefore need a Non residual model
#     augment_model = hybrid_model.augment_model
#     plot_model = Hybrid_Python(reference_variables, augment_model, x_ref_train[0], t_train, mode='trajectory')
#     pred_train = torch.tensor(plot_model())

#     plot_model.t = t_test
#     plot_model.x0 = x0_test
#     pred_test = torch.tensor(plot_model())

#     result_plot_multi_dim('Custom', 'VdP', os.path.join(results_directory, f'Final.png'),
                # t_train, pred_train, t_test, pred_test,
                # np.hstack((t_train, t_test)), np.vstack((x_ref_train, x_ref_test)))

def dump_best_experiment(best_experiment, setup_file, results_file):
    yaml_dict = best_experiment.copy()
    yaml_dict['loss'] = float(best_experiment['loss'])
    yaml_dict['loss_test'] = float(best_experiment['loss_test'])
    del yaml_dict['model']

    with open(setup_file, 'w') as file:
        yaml.dump(yaml_dict, file)

    with open(results_file, 'a') as file:
        file.write(f'Best experiment: {best_experiment["n_exp"]}')
        file.write('\n')

def save_best_experiment(args, hybrid_model, parameter_file='best_parameters.pt'):
    torch.save({
        'epoch': args['epoch'],
        'model_state_dict': hybrid_model.state_dict(),
        'best_loss': args['best_loss'],
        'best_loss_test': args['best_loss_test']
        # 'time': best_experiment['time']
    }, parameter_file)

def get_best_parameters(hybrid_model, parameter_file='best_parameters.pt'):
    ckpt = torch.load(parameter_file)
    hybrid_model.load_state_dict(ckpt['model_state_dict'])
    return hybrid_model, ckpt

def load_parameters(hybrid_model, parameter_file):
    try:
        ckpt = torch.load(parameter_file)
        hybrid_model.load_state_dict(ckpt['model_state_dict'])
        print('Continuing training with previous parameters')
    except:
        print('Could not find model to load')

def load_jax_parameters(hybrid_model, jax_parameters):
    """We assume that the hybrid model and the jax model have the same structure

    Parameters
    ----------
    hybrid_model : _type_
        _description_
    jax_parameters : _type_
        _description_
    """
    jax_parameters_as_list = dict_to_list(jax_parameters)
    jax_parameters_in_torch_format = []
    for i, jax_param in enumerate(jax_parameters_as_list):
        if len(jax_param.shape) == 2:
            jax_param = jax_param.T
        jax_param = torch.tensor(jax_param)
        jax_parameters_in_torch_format.append(jax_param)
    # In JAX/FLAX bias and kernel are swapped; swap them back
    for i in range(0, len(jax_parameters_in_torch_format), 2):
        jax_parameters_in_torch_format[i], jax_parameters_in_torch_format[i+1] = jax_parameters_in_torch_format[i+1], jax_parameters_in_torch_format[i]

    for i, p in enumerate(hybrid_model.parameters()):
        with torch.no_grad():
            p.copy_(jax_parameters_in_torch_format[i])

    return hybrid_model

def dict_to_list(dict_, ret = None):
    if ret is None:
        ret = []
    for k, v in dict_.items():
        if type(v) == dict:
            v = dict_to_list(v, ret)
        else:
            ret.append(v)
    return ret

def sum_loss(x, y):
    a = x-y
    return a.sum()

def train(model:Hybrid_FMU or Hybrid_Python, args: dict):

    pD = args['problemDescription']
    mD = args['modelDescription']
    oD = args['optimizerDescription']
    gS = args['generalSettings']
    residual = args['residual']
    restore = gS['load_parameters']
    reference_data = args['reference_data']
    pointers = args['pointers']
    results_directory = args['results_directory']
    epochs = args['epochs']
    logger = args['logger']
    best_parameter_file = args['best_parameter_file']

    # Optimizes the Neural Network with CBO
    optimizer = Optimizer(model, n_particles=oD['particles'], alpha=oD['alpha'], sigma=oD['sigma'],
                          l=oD['l'], dt=oD['dt'], anisotropic=oD['anisotropic'], eps=oD['eps'], partial_update=oD['partial_update'],
                          use_multiprocessing=gS['use_multiprocessing'], n_processes=gS['processes'],
                          particles_batch_size=oD['particles_batch_size'], device=gS['device'], fmu=pD['fmu'], residual=residual, restore=restore)

    if pD['type'] == 'classification':
        loss_fn = Loss(F.nll_loss, optimizer)
    elif pD['type']  == 'regression':
        loss_fn = Loss(F.mse_loss, optimizer)
        # loss_fn = Loss(sum_loss , optimizer)

    # CALCULATE THE LOSS OVER THE WHOLE TRAJECTORY, NO BATCHES
    t_train = reference_data['t_train']
    t_test = reference_data['t_test']
    x_ref_train = reference_data['x_ref_train']
    x_ref_test = reference_data['x_ref_test']

    model.trajectory_mode()
    model.set_trajectory_variables(x0=x_ref_train[0], t=t_train)
    if isinstance(model, Hybrid_FMU):
        pred_train_traj = torch.tensor(model())
    else:
        pred_train_traj = model()
    model.set_trajectory_variables(x0=x_ref_test[0], t=t_test)
    if isinstance(model, Hybrid_FMU):
        pred_test_traj = torch.tensor(model())
    else:
        pred_test_traj = model()

    if residual:
        for particle in optimizer.particles:
            particle.model.residual_mode()
        model.residual_mode()

    result_plot_multi_dim('Custom', pD['name'], os.path.join(results_directory, f'Prediction Epoch -1.png'),
                t_train, pred_train_traj, t_test, pred_test_traj,
                np.hstack((t_train, t_test)), np.vstack((x_ref_train, x_ref_test)))

    for epoch in range(args['epochs']):
        pred_train_batches = []
        accuracies_train_batches = []
        losses_train_batches = []
        for batch_idx, (input_train, output_train) in enumerate(args['train_dataloader']):
            input_train, output_train = input_train.to(gS['device']), output_train.to(gS['device'])
            # Calculate current solution
            if residual:
                if pD['type'] == 'classification':
                    raise NotImplementedError
                    exit(1)
                else:
                    pred_train_res = model(input_train)
                    if args['epoch'] % gS['eval_freq'] == 0:
                        model.trajectory_mode()
                        t_batch = t_train[input_train.shape[0]*batch_idx:input_train.shape[0]*(batch_idx+1)]
                        x0_batch = x_ref_train[input_train.shape[0]*batch_idx]
                        model.set_trajectory_variables(x0=x0_batch, t=t_batch)
                        if isinstance(model, Hybrid_FMU):
                            pred_train_traj = torch.tensor(model())
                        else:
                            pred_train_traj = model()
                        model.residual_mode()
                        pred_train_batches.append(pred_train_traj)
            else:
                model.t = input_train.detach().numpy()
                model.x0 = output_train[0]
                for particle in optimizer.particles:
                    particle.model.t = input_train.detach().numpy()
                    particle.model.x0 = output_train[0]
                if isinstance(model, Hybrid_FMU):
                    pred_train_traj = torch.tensor(model())
                else:
                    pred_train_traj = torch.tensor(model(stim=False))
                pred_train_batches.append(pred_train_traj.detach().numpy())
                pred_train_res = None

            if residual:
                loss_train = _evaluate_reg(pred_train_res, output_train, F.mse_loss).cpu()
            else:
                loss_train = _evaluate_reg(pred_train_traj, output_train, F.mse_loss).cpu()
            train_acc = 0.0

            accuracies_train_batches.append(train_acc)
            losses_train_batches.append(loss_train.detach().numpy())

            optimizer.zero_grad()
            loss_fn.backward(input_train, output_train, backward_gradients=False)
            optimizer.step()

            if batch_idx % 10 == 0:
                logger.info('Train Epoch: {}/{} [{}/{} ({:.0f}%)]\t Batch Loss: {:.6f}'.format(
                    epoch, args['epochs'], batch_idx * len(input_train), len(args['train_dataloader'].dataset),
                    100. * batch_idx / len(args['train_dataloader']), loss_train.item()))

        if residual:
            # For residuals the loss for all samples is just the mean of the epoch loss
            args['losses_train_res'].append(np.asarray(losses_train_batches).sum())
            # Determine the residual loss for the test dataset
            if pD['type'] == 'classification':
                # loss_train, train_acc = _evaluate_class(model, X, y, F.nll_loss)
                pred_test_res = None
                acc_test_res = None

            else:
                pred_test_res = model(args['test_dataloader'].dataset.x)
                acc_test_res = 0.0
            loss_test_res = _evaluate_reg(pred_test_res, args['test_dataloader'].dataset.y, F.mse_loss).cpu().detach().numpy()
            args['losses_test_res'].append(loss_test_res)
            if args['epoch'] % gS['eval_freq'] == 0:
                pred_train_batched = np.vstack(pred_train_batches)
            else:
                pred_train_batched = None
        else:
            args['losses_train_res'].append(None)
            args['losses_test_res'].append(None)
            pred_train_batched = np.vstack(pred_train_batches)

        args['losses_batches'].append(np.asarray(losses_train_batches).sum())

        ################################################################################
        #### PREDICTION FOR WHOLE TRAJECTORY  ##########################################
        ################################################################################

        model.trajectory_mode()
        model.set_trajectory_variables(x0=x_ref_train[0], t=t_train)
        if isinstance(model, Hybrid_FMU):
            pred_train_traj = torch.tensor(model())
        else:
            pred_train_traj = model()
        model.set_trajectory_variables(x0=x_ref_test[0], t=t_test)
        if isinstance(model, Hybrid_FMU):
            pred_test_traj = torch.tensor(model())
        else:
            pred_test_traj = model()
        if residual:
            model.residual_mode()
        loss_train = _evaluate_reg(torch.Tensor(pred_train_traj), torch.Tensor(x_ref_train), F.mse_loss) * pred_train_traj.shape[0]
        loss_test = _evaluate_reg(torch.Tensor(pred_test_traj), torch.Tensor(x_ref_test), F.mse_loss) * pred_test_traj.shape[0]
        args['losses_train'].append(loss_train)
        args['losses_test'].append(loss_test)
        args['accuracies_train'].append(0)
        args['accuracies_test'].append(0)

        if residual:
            logger.info('Epoch: {:04d}/{}    Loss (TrajTrain): {:.5e}    Loss (TrajTest): {:.5e}    Loss (ResTrain): {:.5e}    Loss (ResTest): {:.5e}   Loss (Batches): {:.5e}'.format(
                            epoch,
                            args['epochs'],
                            args['losses_train'][-1],
                            args['losses_test'][-1],
                            args['losses_train_res'][-1],
                            args['losses_test_res'][-1],
                            args['losses_batches'][-1]
                         ))
        else:
            logger.info(f'Epoch: {epoch:04d}/{args["epochs"]}   Loss (TrajTrain): {args["losses_train"][-1]:.5f}    Loss (TrajTrain, batched): {args["losses_batches"][-1]:.5f}    Loss (TrajTest): {args["losses_test"][-1]:.5f}')

        if epoch % gS['eval_freq'] == 0:
            if gS['plot_prediction']:
                result_plot_multi_dim(mD['name'], pD['name'], os.path.join(results_directory, f'Prediction Epoch {args["epoch"]}.png'),
                            t_train, pred_train_traj, t_test, pred_test_traj,
                            np.hstack((t_train, t_test)), np.vstack((x_ref_train, x_ref_test)))
                if oD['batching']:
                    result_plot_multi_dim(mD['name'], pD['name'], os.path.join(results_directory, f'Prediction (batched) Epoch {args["epoch"]}.png'),
                                t_train[:pred_train_batched.shape[0]], pred_train_batched, t_test, pred_test_traj,
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
                        plot_path=os.path.join(results_directory, 'Loss (Batched).png'),
                        train_acc=args['accuracies_train'],
                        test_acc=args['accuracies_test'],
                        train_loss=args['losses_batches'],
                        test_loss=args['losses_test']
                    )
                if residual:
                    build_plot(epochs=len(args['losses_train_res']),
                        model_name=mD['name'],
                        dataset_name=pD['name'],
                        plot_path=os.path.join(results_directory, 'Loss (Residual).png'),
                        train_acc=args['accuracies_train'],
                        test_acc=args['accuracies_test'],
                        train_loss=args['losses_train_res'],
                        test_loss=args['losses_test_res']
                    )

            if gS['plot_parameters']:
                visualise_wb([p for p in model.parameters()], results_directory, f'Parameters Epoch {args["epoch"]}')

        if args['epoch'] == 0:
            args['best_loss'] = args['losses_test'][-1]
            if gS['save_parameters']:
                save_best_experiment(args, model, best_parameter_file)
        else:
            if args['losses_test'][-1] < args['best_loss']:
                args['best_loss'] = args['losses_test'][-1]
                if gS['save_parameters']:
                    save_best_experiment(args, model, best_parameter_file)

        if oD['cooling']:
            optimizer.cooling_step()

        args['epoch'] += 1
    return

def determine_device(device_type, use_multiprocessing):
    if device_type == 'cuda' and not torch.cuda.is_available():
        print('Cuda is unavailable. Using CPU instead.')
        device_type = 'cpu'
    if device_type != 'cpu' and use_multiprocessing:
        print('Unable to use multiprocessing on GPU')
        use_multiprocessing = False
    return torch.device(device_type), use_multiprocessing

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # PROBLEM SETUP
    parser.add_argument('--dataset', type=str, default='VdP', help='dataset to use',
                        choices=list(DATASETS.keys()))
    parser.add_argument('--problem_type', type=str, choices=['regression', 'classification'], default='regression',
                        help='whether to use GPU (cuda) for accelerated computations or not')
    parser.add_argument('--fmu', type=bool, default=False, help='Whether or not to use the FMU or a python implementation')

    # NUMERICAL PROBLEM SETUP
    parser.add_argument('--start', type=float, default=0.0, help='Start value of the ODE integration')
    parser.add_argument('--end', type=float, default=10.0, help='End value of the ODE integration')
    parser.add_argument('--n_steps', type=int, default=10001, help='How many integration steps to perform')
    parser.add_argument('--ic', type=List, default=[1.0, 0.0], help='initial_condition of the ODE')
    parser.add_argument('--kappa', type=float, default=1.0, help='oscillation constant of the VdP oscillation term')
    parser.add_argument('--mu', type=float, default=3.0, help='damping constant of the VdP damping term')
    parser.add_argument('--mass', type=float, default=1.0, help='mass constant of the VdP system')

    # NEURAL NETWORK SETUP
    parser.add_argument('--model', type=str, default='CustomMLP', help=f'architecture to use',
                        choices=list(MODELS.keys()))
    parser.add_argument('--layers', type=int, default=1, help='Number of hidden layers')
    parser.add_argument('--layer_size', type=int, default=15, help='Number of neurons in a hidden layer')

    # SOLVER SETUP
    parser.add_argument('--residual', required=False, type=bool, default=False,
                        help='Perform Design of Experiments with NN Parameters and ODE Hyperparameters on Residuals')
    parser.add_argument('--trajectory', required=False, type=bool, default=True,
                        help='Perform Design of Experiments with NN Parameters and ODE Hyperparameters on Trajectory')

    # NUMERICAL SOLVER SETUP
    parser.add_argument('--epochs', type=int, default=10, help='train for EPOCHS epochs')
    parser.add_argument('--batch_size', type=int, default=200, help='batch size (for samples-level batching)')
    parser.add_argument('--particles', type=int, default=100, help='')
    parser.add_argument('--particles_batch_size', type=int, default=25, help='batch size '
                                                                             '(for particles-level batching)')
    parser.add_argument('--alpha', type=float, default=50.0, help='alpha from CBO dynamics')
    parser.add_argument('--sigma', type=float, default=1.0 ** 0.5, help='sigma from CBO dynamics')
    parser.add_argument('--l', type=float, default=1.0, help='lambda from CBO dynamics')
    parser.add_argument('--dt', type=float, default=0.1, help='dt from CBO dynamics')
    parser.add_argument('--anisotropic', type=bool, default=True, help='whether to use anisotropic or not')
    parser.add_argument('--eps', type=float, default=1e-5, help='threshold for additional random shift')
    parser.add_argument('--partial_update', type=bool, default=True, help='whether to use partial or full update')
    parser.add_argument('--cooling', type=bool, default=False, help='whether to apply cooling strategy')

    # FILE SETUP
    parser.add_argument('--results_file', type=str, default='doe_results.yaml')
    parser.add_argument('--results_directory', type=str, default=None)

    # MISC. SETUP
    parser.add_argument('--build_plot', required=False, default=True, action='store_true',
                        help='specify to build loss and accuracy plot')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='cuda',
                        help='whether to use GPU (cuda) for accelerated computations or not')
    parser.add_argument('--use_multiprocessing', action='store_true', default=False,
                        help='specify to use multiprocessing for accelerating computations on CPU '
                             '(note, it is impossible to use multiprocessing with GPU)')
    parser.add_argument('--processes', type=int, default=4,
                        help='how many processes to use for multiprocessing')
    parser.add_argument('--eval_freq', type=int, default=5, help='evaluate test accuracy every EVAL_FREQ '
                                                                   'samples-level batches')
    parser.add_argument('--restore', required=False, type=bool, default=False,
                        help='restore previous parameters')
    parser.add_argument('--restore_name', required=False, type=str, default='ckpt')

    args = parser.parse_args()

    path = os.path.abspath(__file__)
    directory = os.path.sep.join(path.split(os.path.sep)[:-1])
    if args.results_directory is None:
        args.results_directory = create_results_directory(directory=directory, results_directory_name='unnamed_CBO')

    log_file = os.path.join(args.results_directory, 'CBO.log')
    logging.basicConfig(filename=log_file, encoding='utf-8', level=logging.INFO)
    logger = logging.getLogger('CBO')

    if args.batch_size == -1:
        args.batch_size = args.n_steps
        if args.residual:
            # For residuals we have one sample less available because of the finite difference calculation
            args.batch_size -= 1

    device, use_multiprocessing = determine_device(args.device, args.use_multiprocessing)

    # CREATE REFERENCE SOLUTION
    logger.info('Creating reference solution')
    ic = np.array(args.ic)
    reference_variables = [args.kappa, args.mu, args.mass]
    t_train, x_ref_train, t_test, x_ref_test, x0_test = create_reference_solution(args.start, args.end, args.n_steps, ic, reference_variables, ode_integrator=euler)
    train_residual_inputs, train_residual_outputs, test_residual_inputs, test_residual_outputs = create_residual_reference_solution(t_train, x_ref_train, t_test, x_ref_test, reference_variables)

    # PUT THE REFERENCE SOLUTION IN A CONTAINER FOR EASIER ACCESS
    plotting_reference_data = {'t_train': t_train,
                               't_test': t_test,
                               'x_ref_train': x_ref_train,
                               'x_ref_test': x_ref_test,
                               'results_directory': args.results_directory}

    if args.restore:
        restore_directory = os.path.join(directory, args.restore_name)
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=5, create=True)
        checkpoint_manager = orbax.checkpoint.CheckpointManager(restore_directory, orbax_checkpointer, options)
        step = checkpoint_manager.latest_step()
        jax_parameters = checkpoint_manager.restore(step)

    if args.fmu:
        logger.error('FMU not yet implemented')
        raise NotImplementedError

    else:
        if args.residual:
            logger.info('Preparing residual experiment.')
            # CONVERT THE REFERENCE DATA TO A DATASET
            train_dataset = create_generic_dataset(torch.tensor(train_residual_inputs), torch.tensor(train_residual_outputs))
            test_dataset = create_generic_dataset(torch.tensor(test_residual_inputs), torch.tensor(test_residual_outputs))
            train_dataloader, test_dataloader = load_generic_dataloaders(train_dataset=train_dataset,
                                                                        train_batch_size=args.batch_size if args.batch_size < len(train_dataset) else len(train_dataset),
                                                                        test_dataset=test_dataset,
                                                                        test_batch_size=args.batch_size if args.batch_size < len(test_dataset) else len(test_dataset),
                                                                        shuffle=False)
            start = time.time()
            residual_directory, checkpoint_directory = create_results_subdirectories(results_directory=args.results_directory, trajectory=False, residual=True)
            plotting_reference_data['results_directory'] = residual_directory

            layers = layers = [2] + [args.layer_size]*args.layers + [1]
            augment_model = CustomMLP(layers)
            hybrid_model = Hybrid_Python(reference_variables, augment_model, ic, t_train, mode='residual')

            if args.restore:
                hybrid_model = load_jax_parameters(hybrid_model, jax_parameters)

            logger.info('Training...')
            result = train(model=hybrid_model,
                           train_dataloader=train_dataloader,
                           test_dataloader=test_dataloader,
                           device=device,
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
                           eval_freq=args.eval_freq,
                           problem_type=args.problem_type,
                           pointers=None,
                           residual=True,
                           plotting_reference_data=plotting_reference_data,
                           restore=args.restore,
                           logger=logger)

            experiment_time = time.time() - start
            accuracies_train, accuracies_test, losses_train, losses_test = result
            hybrid_model, ckpt = get_best_parameters(hybrid_model, residual_directory)

            logger.info(f'Best Epoch ({ckpt["epoch"]}/{args.epochs})- Training loss: {ckpt["loss"]:3.5f}, Validation loss: {ckpt["loss_test"]:3.5f}, Time: {experiment_time:3.3f}')

            if args.build_plot:
                # augment_model = hybrid_model.augment_model
                # plot_model = Hybrid_Python(reference_variables, augment_model, x_ref_train[0], t_train, mode='trajectory')
                hybrid_model.t = t_train
                hybrid_model.x0 = ic
                hybrid_model.trajectory_mode()
                pred_train = hybrid_model()
                hybrid_model.t = t_test
                hybrid_model.x0 = x0_test
                pred_test = hybrid_model()
                build_plot(args.epochs, args.model, args.dataset, os.path.join(args.results_directory, 'Loss.png'), *result)
                result_plot_multi_dim(args.model, args.dataset, os.path.join(plotting_reference_data['results_directory'], f'Final.png'),
                                      t_train, pred_train, t_test, pred_test,
                                      np.hstack((t_train, t_test)), np.vstack((x_ref_train, x_ref_test)))
                # plot_loss_and_prediction(result, hybrid_model, args, reference_variables, plotting_reference_data)

            results_dict = {
                'accuracies_train': list(accuracies_train),
                'accuracies_test': list(accuracies_test),
                'losses_train': list(losses_train),
                'losses_test': list(losses_test),
                'time': experiment_time
            }

            logger.info(f'Dumping results to {args.results_file}.')
            with open(args.results_file, 'w') as file:
                yaml.dump(results_dict, file)

        if args.trajectory:
            logger.info('Preparing trajectory experiment.')
            # CONVERT THE REFERENCE DATA TO A DATASET
            train_dataset = create_generic_dataset(torch.tensor(t_train), torch.tensor(x_ref_train))
            test_dataset = create_generic_dataset(torch.tensor(t_test), torch.tensor(x_ref_test))

            train_dataloader, test_dataloader = load_generic_dataloaders(train_dataset=train_dataset,
                                                                        train_batch_size=args.batch_size if args.batch_size < len(train_dataset) else len(train_dataset),
                                                                        test_dataset=test_dataset,
                                                                        test_batch_size=args.batch_size if args.batch_size < len(test_dataset) else len(test_dataset),
                                                                        shuffle=False)

            start = time.time()
            trajectory_directory, checkpoint_directory = create_results_subdirectories(results_directory=args.results_directory, trajectory=True, residual=False)
            plotting_reference_data['results_directory'] = trajectory_directory

            layers = layers = [2] + [args.layer_size]*args.layers + [1]
            augment_model = CustomMLP(layers)
            hybrid_model = Hybrid_Python(reference_variables, augment_model, ic, t_train, mode='trajectory')

            result = train(model=hybrid_model,
                           train_dataloader=train_dataloader,
                           test_dataloader=test_dataloader,
                           device=device,
                           use_multiprocessing=args.use_multiprocessing,
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
                           eval_freq=args.eval_freq,
                           problem_type='regression',
                           pointers=None,
                           residual=False,
                           plotting_reference_data=plotting_reference_data,
                           restore=args.restore,
                           logger=logger)
            experiment_time = time.time() - start
            accuracies_train, accuracies_test, losses_train, losses_test = result
            hybrid_model, ckpt = get_best_parameters(hybrid_model, trajectory_directory)

            logger.info(f'Best Epoch ({ckpt["epoch"]}/{args.epochs})- Training loss: {ckpt["loss"]:3.5f}, Validation loss: {ckpt["loss_test"]:3.5f}, Time: {experiment_time:3.3f}')

            if args.build_plot:
                hybrid_model.t = t_train
                hybrid_model.x0 = ic
                hybrid_model.trajectory_mode()
                pred_train = hybrid_model()
                hybrid_model.t = t_test
                hybrid_model.x0 = x0_test
                pred_test = hybrid_model()
                build_plot(args.epochs, args.model, args.dataset, os.path.join(args.results_directory, 'Loss.png'), *result)
                result_plot_multi_dim(args.model, args.dataset, os.path.join(plotting_reference_data['results_directory'], f'Final.png'),
                                      t_train, pred_train, t_test, pred_test,
                                      np.hstack((t_train, t_test)), np.vstack((x_ref_train, x_ref_test)))

                # plot_loss_and_prediction(result, hybrid_model, args, reference_variables, plotting_reference_data)

            results_dict = {
                'accuracies_train': list(accuracies_train),
                'accuracies_test': list(accuracies_test),
                'losses_train': list(losses_train),
                'losses_test': list(losses_test),
                'time': experiment_time
            }

            logger.info(f'Dumping results to {args.results_file}.')
            with open(args.results_file, 'w') as file:
                yaml.dump(results_dict, file)

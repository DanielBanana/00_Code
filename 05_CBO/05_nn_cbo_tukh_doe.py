import os
import sys
import shutil
import torch
import torch.nn.functional as F
import numpy as np
import argparse
import time
import matplotlib.pyplot as plt

import jax
from jax import lax, jit
import jax.numpy as jnp
from functools import partial

sys.path.insert(1, os.getcwd())
from plot_results import plot_results, plot_losses, get_file_path
from fmu_helper import FMUEvaluator
from cbo_in_python.src.torch_.models import *
# from cbo_in_python.src.datasets import load_mnist_dataloaders, load_parabola_dataloaders, f
from cbo_in_python.src.datasets import create_generic_dataset, load_generic_dataloaders
from cbo_in_python.src.torch_.optimizer import Optimizer
from cbo_in_python.src.torch_.loss import Loss
from torch.utils.data import Dataset, DataLoader

from collections import OrderedDict
from pyDOE2 import fullfact
import datetime

MODELS = {
    'SimpleMLP': SimpleMLP,
    'TinyMLP': TinyMLP,
    'SmallMLP': SmallMLP,
    'LeNet1': LeNet1,
    'LeNet5': LeNet5,
}

DATASETS = {
    'VdP': ''
}

# The Neural Network structure class
class VdPMLP(nn.Module):
    def __init__(self):
        super(VdPMLP, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(2, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),

            nn.Linear(20, 1)
        )
        self.double()

    def forward(self, x):
        return self.model(x)

# Managing class for Hybrid model with FMU
class Hybrid_FMU(nn.Module):
    def __init__(self, fmu_model, augment_model, z0, t):
        super(Hybrid_FMU, self).__init__()

        self.fmu_model = fmu_model
        self.augment_model = augment_model
        self.z0 = z0
        self.t = t

    def augment_model_function(self, augment_model_parameters, input):
        # The augment_model is currently a pytorch model, which just takes
        # the input. It has its own parameters saved internally.
        # The f_euler function expects a model which needs its paramaters
        # given when it is called: y = augment_model_function(parameters, input)
        # f_euler provides the input to the augment_model as numpy array
        # but we can only except tensors, so convert
        return self.augment_model(torch.tensor(input)).detach().numpy()

    def forward(self, pointers):
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

            pointers, enterEventMode, terminateSimulation = self.fmu_model.evaluate_fmu(t[i], dt, self.augment_model_function, None, pointers)

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
class Hybrid_Python(nn.Module):
    def __init__(self, augment_model, z0, t):
        super(Hybrid_Python, self).__init__()

        self.ode_parameters = [1.0, 8.53, 1.0]
        self.augment_model = augment_model
        self.z0 = z0
        self.t = t

    def augment_model_function(self, augment_model_parameters, input):
        # The augment_model is currently a pytorch model, which just takes
        # the input. It has its own parameters saved internally.
        # The f_euler function expects a model which needs its paramaters
        # given when it is called: y = augment_model_function(parameters, input)
        # f_euler provides the input to the augment_model as numpy array
        # but we can only except tensors, so convert
        return self.augment_model(torch.tensor(input)).detach().numpy()

    def forward(self, stim=False):
        t = self.t
        z0 = self.z0
        z = np.zeros((t.shape[0], z0.shape[0]))
        z[0] = z0
        # Forward the initial state to the FMU
        times = []
        if stim:
            self.hybrid_ode = self.hybrid_ode_stim
        else:
            # NN parameters are saved inside the augment_model and therefore not needed as input
            self.hybrid_ode = self.hybrid_ode
            z = self.hybrid_euler(ode_parameters=self.ode_parameters, nn_parameters=None)
        return z

    def hybrid_euler(self, ode_parameters, nn_parameters):
        '''Applies forward Euler to the hybrid ODE and returns the trajectory'''
        t = self.t
        z0 = self.z0
        z = np.zeros((t.shape[0], z0.shape[0]))
        # z = z.at[0].set(z0)
        z[0] = z0
        i = np.asarray(range(t.shape[0]))
        # We can replace the loop over the time by a lax.scan this is 3 times as fast: e.g.: 0.32-0.26 -> 0.11-0.9
        # euler_body_func = partial(self.hybrid_step, t=t, ode_parameters=ode_parameters, nn_parameters=nn_parameters)
        # final, result = lax.scan(euler_body_func, z0, i)
        # z = z.at[1:].set(result[:-1])
        for i in range(len(t)-1):
            dt = t[i+1] - t[i]
            z[i+1] = z[i] + dt * self.hybrid_ode(z[i], t[i], ode_parameters, nn_parameters)
        return z

    def hybrid_ode(self, z, t, ode_parameters, nn_parameters):
        '''Calculates the right hand side of the hybrid ODE, where
        the damping term is replaced by the neural network'''
        kappa = ode_parameters[0]
        mu = ode_parameters[1]
        mass = ode_parameters[2]
        derivative = np.array([np.array((z[1],)),
                                np.array((-kappa*z[0]/mass,)) + self.augment_model_function(nn_parameters, z)]).flatten()
        return derivative

    def hybrid_ode_stim(self, z, t, ode_parameters, nn_parameters):
        '''Calculates the right hand side of the hybrid ODE, where
        the damping term is replaced by the neural network'''
        kappa = ode_parameters[0]
        mu = ode_parameters[1]
        mass = ode_parameters[2]
        derivative = np.array([np.array((z[1],)),
                                np.array((-kappa*z[0]/mass,)) + self.augment_model_function(nn_parameters, z) + np.array(1.2*np.cos(np.pi/5*t))]).flatten()
        return derivative


class Hybrid_Python_Residual(nn.Module):
    def __init__(self, augment_model, z0, t):
        super(Hybrid_Python_Residual, self).__init__()

        self.ode_parameters = [1.0, 8.53, 1.0]
        self.augment_model = augment_model
        self.z0 = z0
        self.t = t

    def augment_model_function(self, augment_model_parameters, input):
        # The augment_model is currently a pytorch model, which just takes
        # the input. It has its own parameters saved internally.
        # The f_euler function expects a model which needs its paramaters
        # given when it is called: y = augment_model_function(parameters, input)
        # f_euler provides the input to the augment_model as numpy array
        # but we can only except tensors, so convert
        return self.augment_model(torch.tensor(input))

    def forward(self, input):
        return augment_model(input)

# FOR JAX.LAX probably not sure if compatible with torch
# def hybrid_step(prev_z, i, t, ode_parameters, nn_parameters):
#     t = jnp.asarray(t)
#     dt = t[i+1] - t[i]
#     next_z = prev_z + dt * hybrid_ode(prev_z, t[i], ode_parameters, nn_parameters)
#     return next_z, next_z

# For calculation of the reference solution we need the correct behaviour of the VdP
def damping(mu, inputs):
    return mu * (1 - inputs[0]**2) * inputs[1]

def _evaluate_reg(outputs, y_, loss_fn):
    with torch.no_grad():
        loss = loss_fn(outputs, y_)
    return loss

def train(hybrid_model:Hybrid_FMU or Hybrid_Python, train_dataloader, val_dataloader, device, use_multiprocessing, processes,
          epochs, particles, particles_batch_size,
          alpha, sigma, l, dt, anisotropic, eps, partial_update, cooling,
          eval_freq, problem_type, pointers):
    train_accuracies = []
    train_losses = []
    val_accuracies = []
    val_losses = []

    # Optimizes the Neural Network with CBO
    augment_optimizer = Optimizer(hybrid_model, n_particles=particles, alpha=alpha, sigma=sigma,
                          l=l, dt=dt, anisotropic=anisotropic, eps=eps, partial_update=partial_update,
                          use_multiprocessing=use_multiprocessing, n_processes=processes,
                          particles_batch_size=particles_batch_size, device=device)

    augment_model_parameters = []

    if problem_type == 'classification':
        pass
        # loss_fn = Loss(F.nll_loss, optimizer)
    else:
        # loss_fn = Loss(F.mse_loss, augment_optimizer)
        loss_fn = Loss(F.mse_loss, augment_optimizer)

    n_batches = len(train_dataloader)

    for epoch in range(epochs):
        epoch_train_accuracies = []
        epoch_train_losses = []
        for batch, (X, y) in enumerate(train_dataloader):
            X_train, y_train = X.to(device), y.to(device)

            # Calculate current solution
            hybrid_model.t = X_train.detach().numpy()
            hybrid_model.z0 = y_train[0]
            if isinstance(hybrid_model, Hybrid_FMU):
                z = torch.tensor(hybrid_model(pointers))
            else:
                z = torch.tensor(hybrid_model(stim=False))

            if problem_type == 'classification':
                pass
                # train_loss, train_acc = _evaluate_class(model, X, y, F.nll_loss)
            else:
                train_loss = _evaluate_reg(z, y_train, F.mse_loss)
                train_acc = 0.0
            epoch_train_accuracies.append(train_acc)
            epoch_train_losses.append(train_loss.cpu())

            augment_optimizer.zero_grad()
            loss_fn.backward(z, y, backward_gradients=False)
            augment_optimizer.step()

            if batch % eval_freq == 0 or batch == n_batches - 1:
                with torch.no_grad():
                    losses = []
                    accuracies = []
                    for X_val, y_val in val_dataloader:
                        X_val, y_val = X_val.to(device), y_val.to(device)
                        if problem_type == 'classification':
                            pass
                            # loss, acc = _evaluate_class(model, X_val, y_val, F.nll_loss)
                        else:
                            hybrid_model.z0 = y_val[0]
                            hybrid_model.t = X_val.detach().numpy()
                            if type(hybrid_model)==Hybrid_FMU:
                                z = torch.tensor(hybrid_model(pointers))
                            else:
                                z = torch.tensor(hybrid_model(stim=False))
                            loss = _evaluate_reg(z, y_val, F.mse_loss)
                            acc = 0.0
                            losses.append(loss.cpu())
                            accuracies.append(acc)
                    val_loss, val_acc = np.mean(losses), np.mean(accuracies)
                    if batch == n_batches - 1:
                        val_accuracies.append(val_acc)
                        val_losses.append(val_loss)

            print(
                f'Epoch: {epoch + 1:2}/{epochs}, batch: {batch + 1:4}/{n_batches}, train loss: {train_loss:8.3f}, '
                f'train acc: {train_acc:8.3f}, val loss: {val_loss:8.3f}, val acc: {val_acc:8.3f}, alpha: {augment_optimizer.alpha:8.3f}, sigma: {augment_optimizer.sigma:8.3f}',
                flush=True)

        train_accuracies.append(np.mean(epoch_train_accuracies))
        train_losses.append(np.mean(epoch_train_losses))
        if cooling:
            augment_optimizer.cooling_step()

    return train_accuracies, val_accuracies, train_losses, val_losses

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



def build_plot(epochs, model_name, dataset_name, plot_path,
               train_acc, val_acc, train_loss, val_loss):
    plt.rcParams['figure.figsize'] = (20, 10)
    plt.rcParams['font.size'] = 25

    epochs_range = np.arange(1, epochs + 1, dtype=int)

    plt.clf()
    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.plot(epochs_range, train_acc, label='train')
    ax1.plot(epochs_range, val_acc, label='val')
    ax1.legend()
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('accuracy')
    ax1.set_title('Accuracy')

    ax2.plot(epochs_range, train_loss, label='train')
    ax2.plot(epochs_range, val_loss, label='val')
    ax2.legend()
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('loss')
    ax2.set_title('Loss')

    plt.suptitle(f'{model_name} @ {dataset_name}')
    plt.savefig(plot_path)

def result_plot(model_name, dataset_name, plot_path,
                X_train, y_train, X_val, y_val, X_reference, y_reference):
    plt.rcParams['figure.figsize'] = (20, 10)
    plt.rcParams['font.size'] = 25

    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.plot(X_reference, y_reference, label='ref')
    ax1.scatter(X_train, y_train, label='train')
    ax1.legend()
    ax1.set_xlabel('X')
    ax1.set_ylabel('y')
    ax1.set_title('Train')

    ax2.plot(X_reference, y_reference, label='ref')
    ax2.scatter(X_val, y_val, label='val')
    ax2.legend()
    ax2.set_xlabel('X')
    ax2.set_ylabel('y')
    ax2.set_title('Test')

    plt.suptitle(f'{model_name} @ {dataset_name}')
    plt.savefig(plot_path)

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

def create_results_subdirectories(results_directory, t=False, r=False, c=True):
    if t:
        trajectory_directory = os.path.join(results_directory, 'trajectory')
        if not os.path.exists(trajectory_directory):
            os.mkdir(trajectory_directory)
    else:
        trajectory_directory = None

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

    return trajectory_directory, residual_directory, checkpoint_directory


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', type=str, default='SimpleMLP', help=f'architecture to use',
                        choices=list(MODELS.keys()))
    parser.add_argument('--dataset', type=str, default='VdP', help='dataset to use',
                        choices=list(DATASETS.keys()))
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='cuda',
                        help='whether to use GPU (cuda) for accelerated computations or not')
    parser.add_argument('--use_multiprocessing', action='store_true',
                        help='specify to use multiprocessing for accelerating computations on CPU '
                             '(note, it is impossible to use multiprocessing with GPU)')
    parser.add_argument('--processes', type=int, default=4,
                        help='how many processes to use for multiprocessing')

    parser.add_argument('--epochs', type=int, default=100, help='train for EPOCHS epochs')
    parser.add_argument('--batch_size', type=int, default=20, help='batch size (for samples-level batching)')
    parser.add_argument('--particles', type=int, default=100, help='')
    parser.add_argument('--particles_batch_size', type=int, default=20, help='batch size '
                                                                             '(for particles-level batching)')
    parser.add_argument('--stimulate', type=bool, default=False, help='Stimulate the VdP or not')

    parser.add_argument('--alpha', type=float, default=100, help='alpha from CBO dynamics')
    parser.add_argument('--sigma', type=float, default=0.1 ** 0.5, help='sigma from CBO dynamics')
    parser.add_argument('--l', type=float, default=1, help='lambda from CBO dynamics')
    parser.add_argument('--dt', type=float, default=0.1, help='dt from CBO dynamics')
    parser.add_argument('--anisotropic', type=bool, default=True, help='whether to use anisotropic or not')
    parser.add_argument('--eps', type=float, default=1e-4, help='threshold for additional random shift')
    parser.add_argument('--partial_update', type=bool, default=True, help='whether to use partial or full update')
    parser.add_argument('--cooling', type=bool, default=True, help='whether to apply cooling strategy')

    parser.add_argument('--build_plot', required=False, default=True, action='store_true',
                        help='specify to build loss and accuracy plot')
    parser.add_argument('--plot_path', required=False, type=str, default='demo.png',
                        help='path to save the resulting plot')

    parser.add_argument('--eval_freq', type=int, default=100, help='evaluate val accuracy every EVAL_FREQ '
                                                                   'samples-level batches')
    parser.add_argument('--fmu', type=bool, default=False, help='Whether or not to use the FMU or a python implementation')
    parser.add_argument('--doe_trajectory', type=bool, default=False, help='Whether or not to use the FMU or a python implementation')
    parser.add_argument('--doe_residual', required=False, type=bool, default=True,
                        help='Perform Design of Experiments with NN Parameters and ODE Hyperparameters on Residuals')


    parser.add_argument('--restore', required=False, type=bool, default=False,
                        help='restore previous parameters')
    parser.add_argument('--overwrite', required=False, type=bool, default=True,
                        help='overwrite previous result')
    parser.add_argument('--results_name', required=False, type=str, default='plot_val',
                        help='name under which the results should be saved, like plots and such, ignored for DoE')
    parser.add_argument('--start', type=float, default=0.0, help='Start value of the ODE integration')
    parser.add_argument('--end', type=float, default=10.0, help='End value of the ODE integration')
    parser.add_argument('--n_steps', type=float, default=1001, help='How many integration steps to perform')


    args = parser.parse_args()

    path = os.path.abspath(__file__)
    directory = os.path.sep.join(path.split(os.path.sep)[:-1])
    file_path = get_file_path(path)

    if args.fmu:
        # ODE SETUP
        ####################################################################################
        # Training Setup
        train_Tstart = 0.0
        train_Tend = 100.0
        train_nSteps = 5001
        train_t = np.linspace(train_Tstart, train_Tend, train_nSteps)
        train_z0 = np.array([1.0, 0.0])

        # Test Setup
        val_Tstart = train_Tend
        val_Tend = train_Tend + (train_Tend - train_Tstart)*0.5
        val_nSteps = int(train_nSteps * 0.5)
        val_t = np.linspace(val_Tstart, val_Tend, val_nSteps)

        mu = 5.0

        # FMU SETUP
        ####################################################################################
        fmu_filename = 'Van_der_Pol_damping_input.fmu'
        path = os.path.abspath(__file__)
        fmu_filename = '/'.join(path.split('/')[:-1]) + '/' + fmu_filename
        fmu_evaluator = FMUEvaluator(fmu_filename, train_Tstart, train_Tend)
        pointers = fmu_evaluator.get_pointers()

        train_z = f_euler_fmu(z0=train_z0, t=train_t, fmu_evaluator=fmu_evaluator, model=damping, model_parameters=mu, pointers=pointers)
        fmu_evaluator.reset_fmu(val_Tstart, val_Tend)
        val_z = f_euler_fmu(z0=train_z[-1], t=val_t, fmu_evaluator=fmu_evaluator, model=damping, model_parameters=mu, pointers=pointers)
        fmu_evaluator.reset_fmu(train_Tstart, train_Tend)

        # CONVERT THE REFERENCE DATA TO A DATASET
        ####################################################################################

        train_dataset = create_generic_dataset(torch.tensor(train_t), torch.tensor(train_z))
        val_dataset = create_generic_dataset(torch.tensor(val_t), torch.tensor(val_z))

        train_dataloader, val_dataloader = load_generic_dataloaders(train_dataset=train_dataset,
                                                                    train_batch_size=train_nSteps,
                                                                    val_dataset=val_dataset,
                                                                    val_batch_size=val_nSteps)

        # TORCH DEVICES
        ####################################################################################
        device = args.device
        if args.device == 'cuda' and not torch.cuda.is_available():
            print('Cuda is unavailable. Using CPU instead.')
            device = 'cpu'
        use_multiprocessing = args.use_multiprocessing
        if device != 'cpu' and use_multiprocessing:
            print('Unable to use multiprocessing on GPU')
            use_multiprocessing = False
        device = torch.device(device)


        # TRAINING
        ####################################################################################
        augment_model = VdPMLP()
        hybrid_model = Hybrid_FMU(fmu_evaluator, augment_model, train_z0, train_t)
        start_time = time.time()
        result = train(hybrid_model, train_dataloader, val_dataloader, device, use_multiprocessing, args.processes,
                    args.epochs, args.particles, args.particles_batch_size,
                    args.alpha, args.sigma, args.l, args.dt, args.anisotropic, args.eps, args.partial_update,
                    args.cooling,
                    args.eval_freq,
                    problem_type='regression',
                    pointers=pointers)
        print(f'Elapsed time: {time.time() - start_time} seconds')
        if args.build_plot:
            build_plot(args.epochs, args.model, args.dataset, 'loss_' + args.plot_path,
                    *result)

            X_train = train_dataloader.dataset.x
            X_val = val_dataloader.dataset.x
            hybrid_model.t = X_train.detach().numpy()
            hybrid_model.z0 = train_dataloader.dataset.y[0]
            z_train = torch.tensor(hybrid_model(pointers))

            hybrid_model.z0 = val_dataloader.dataset.y[0]
            hybrid_model.t = X_val.detach().numpy()
            z_val = torch.tensor(hybrid_model(pointers))
            result_plot(args.model, args.dataset, 'predictions_' + args.plot_path, X_train,
                        z_train[:,0], X_val,
                        z_val[:,0], np.hstack((train_t, val_t)), np.vstack((train_z, val_z))[:,0])
            result_plot(args.model, args.dataset, 'predictions_' + args.plot_path, X_train,
                        z_train[:,1], X_val,
                        z_val[:,1], np.hstack((train_t, val_t)), np.vstack((train_z, val_z))[:,1])

    else:
        if args.doe_residual:
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
            print('DoE currently does not work but residual optimization is worked on')
            start = time.time()
            f_euler = f_euler_python
            results_directory = create_results_directory(directory=directory,
                                                         results_name=args.results_name,
                                                         restore=args.restore,
                                                         overwrite=args.overwrite)
            with open(os.path.join(results_directory, 'Arguments'), 'a') as file:
                file.write(str(args))

            _, residual_directory, checkpoint_directory = create_results_subdirectories(results_directory=results_directory, a=False, r=True)

            reference_ode_parameters = np.asarray([args.kappa, args.mu, args.mass])
            reference_ode_parameters = np.asarray([1.0, 8.53, 1.0])

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

                _, residual_directory, checkpoint_directory = create_results_subdirectories(experiment_directory, r=True)

                z0 = np.array([1.0, 0.0])

                t_train = np.linspace(args.start, args.end, args.n_steps)
                z_ref = np.asarray(f_euler(z0, t_train, reference_ode_parameters))

                # Generate the reference data for validation
                z0_val = z_ref[-1]
                n_steps_val = int(args.n_steps * 0.5)
                t_val = np.linspace(args.end, (args.end-args.start) * 1.5, n_steps_val)
                z_ref_val = np.asarray(f_euler(z0_val, t_val, reference_ode_parameters))

                train_residual_outputs = create_residual_references(z_ref, t_train, reference_ode_parameters)[:,1]
                train_residual_inputs = z_ref[:-1]

                val_residual_outputs = create_residual_references(z_ref_val, t_val, reference_ode_parameters)[:,1]
                val_residual_inputs = z_ref_val[:-1]

                # CONVERT THE REFERENCE DATA TO A DATASET
                ####################################################################################
                train_dataset = create_generic_dataset(torch.tensor(train_residual_inputs), torch.tensor(train_residual_outputs))
                val_dataset = create_generic_dataset(torch.tensor(val_residual_inputs), torch.tensor(val_residual_outputs))

                train_dataloader, val_dataloader = load_generic_dataloaders(train_dataset=train_dataset,
                                                                            train_batch_size=args.batch_size,
                                                                            val_dataset=val_dataset,
                                                                            val_batch_size=n_steps_val)

                # TORCH DEVICES
                ####################################################################################
                device = args.device
                if args.device == 'cuda' and not torch.cuda.is_available():
                    print('Cuda is unavailable. Using CPU instead.')
                    device = 'cpu'
                use_multiprocessing = args.use_multiprocessing
                if device != 'cpu' and use_multiprocessing:
                    print('Unable to use multiprocessing on GPU')
                    use_multiprocessing = False
                device = torch.device(device)
                # TRAINING
                ####################################################################################
                augment_model = VdPMLP()
                hybrid_model = Hybrid_Python_Residual(augment_model, z0, t_train)

        if args.doe_trajectory:
            print('Design of Experiment for CBO not yet implemented')
        else:
            f_euler = f_euler_python
            results_directory = create_results_directory(directory=directory,
                                                         results_name=args.results_name,
                                                         restore=args.restore,
                                                         overwrite=args.overwrite)

            with open(os.path.join(results_directory, 'Arguments'), 'a') as file:
                file.write(str(args))

            trajectory_directory, _, checkpoint_directory = create_results_subdirectories(results_directory=results_directory, t=True, r=False)

            # checkpoint_manager = create_checkpoint_manager(checkpoint_directory=checkpoint_directory,
            #                                                max_to_keep=5)

            reference_ode_parameters = np.asarray([1.0, 8.53, 1.0])

            # Generate the reference data for training
            if args.stimulate:
                ode = ode_stim
                ode_res = ode_stim_res
                z0 = np.array([1.0, 0.0])
            else:
                z0 = np.array([1.0, 0.0])

            # train_z = f_euler_fmu(z0=train_z0, t=train_t, fmu_evaluator=fmu_evaluator, model=damping, model_parameters=mu, pointers=pointers)
            # fmu_evaluator.reset_fmu(val_Tstart, val_Tend)
            # val_z = f_euler_fmu(z0=train_z[-1], t=val_t, fmu_evaluator=fmu_evaluator, model=damping, model_parameters=mu, pointers=pointers)
            # fmu_evaluator.reset_fmu(train_Tstart, train_Tend)


            t_train = np.linspace(args.start, args.end, args.n_steps)
            z_ref = np.asarray(f_euler(z0, t_train, reference_ode_parameters))

            # Generate the reference data for validation
            z0_val = z_ref[-1]
            n_steps_val = int(args.n_steps * 0.5)
            t_val = np.linspace(args.end, (args.end-args.start) * 1.5, n_steps_val)
            z_ref_val = np.asarray(f_euler(z0_val, t_val, reference_ode_parameters))

            # CONVERT THE REFERENCE DATA TO A DATASET
            ####################################################################################
            train_dataset = create_generic_dataset(torch.tensor(t_train), torch.tensor(z_ref))
            val_dataset = create_generic_dataset(torch.tensor(t_val), torch.tensor(z_ref_val))

            train_dataloader, val_dataloader = load_generic_dataloaders(train_dataset=train_dataset,
                                                                         train_batch_size=args.n_steps,
                                                                         val_dataset=val_dataset,
                                                                         val_batch_size=n_steps_val)

            # TORCH DEVICES
            ####################################################################################
            device = args.device
            if args.device == 'cuda' and not torch.cuda.is_available():
                print('Cuda is unavailable. Using CPU instead.')
                device = 'cpu'
            use_multiprocessing = args.use_multiprocessing
            if device != 'cpu' and use_multiprocessing:
                print('Unable to use multiprocessing on GPU')
                use_multiprocessing = False
            device = torch.device(device)

            # TRAINING
            ####################################################################################
            augment_model = VdPMLP()
            hybrid_model = Hybrid_Python(augment_model, z0, t_train)
            start_time = time.time()
            result = train(hybrid_model, train_dataloader, val_dataloader, device, use_multiprocessing, args.processes,
                        args.epochs, args.particles, args.particles_batch_size,
                        args.alpha, args.sigma, args.l, args.dt, args.anisotropic, args.eps, args.partial_update,
                        args.cooling,
                        args.eval_freq,
                        problem_type='regression',
                        pointers=None)
            print(f'Elapsed time: {time.time() - start_time} seconds')
            if args.build_plot:
                build_plot(args.epochs, args.model, args.dataset, 'loss_' + args.plot_path,
                        *result)

                X_train = train_dataloader.dataset.x
                X_val = val_dataloader.dataset.x
                hybrid_model.t = X_train.detach().numpy()
                hybrid_model.z0 = train_dataloader.dataset.y[0]
                z_train = torch.tensor(hybrid_model())

                hybrid_model.z0 = val_dataloader.dataset.y[0]
                hybrid_model.t = X_val.detach().numpy()
                z_val = torch.tensor(hybrid_model())
                result_plot(args.model, args.dataset, 'predictions_' + args.plot_path, X_train,
                            z_train[:,0], X_val,
                            z_val[:,0], np.hstack((t_train, t_val)), np.vstack((z_ref, z_ref_val))[:,0])
                result_plot(args.model, args.dataset, 'predictions_' + args.plot_path, X_train,
                            z_train[:,1], X_val,
                            z_val[:,1], np.hstack((t_train, t_val)), np.vstack((z_ref, z_ref_val))[:,1])
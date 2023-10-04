
# For interaction with the FMU
# import fmpy
# from fmpy import read_model_description, extract
# from fmpy.fmi2 import _FMU2 as FMU2
# import ctypes
# from types import SimpleNamespace
# from typing import List


from fmu_helper import FMUEvaluator

# For automatic differentiaton
import jax
from jax import random, jit, flatten_util, numpy as jnp
from flax import linen as nn
from flax.core import unfreeze, freeze
from typing import Sequence

# For optimisation
from scipy.optimize import minimize

# General
import numpy as np
from matplotlib import pyplot as plt
import os
import sys
import time
from functools import partial
from create_NN_FMU import create_NN_FMU

# To use the plot_results file we need to add the uppermost folder to the PYTHONPATH
# Only Works if file gets called from 00_Code
sys.path.insert(0, os.getcwd())
from plot_results import plot_results, get_file_path
from jax.config import config
config.update("jax_debug_nans", False)
config.update("jax_enable_x64", True)

@jit
def adjoint_f(adj, z, z_ref, t, optimisation_parameters, df_dz_at_t):
    '''Calculates the right hand side of the adjoint system.'''
    dg_dz = jax.grad(g, argnums=0)(z, z_ref, optimisation_parameters)
    d_adj = - df_dz_at_t.T @ adj - dg_dz
    return d_adj

def df_dtheta_function(df_dinput, dinput_dtheta):
    """Calculate the jacobian of the hybrid function (FMU + ML) with respect to the ML
    parameters

    Parameters
    ----------
    df_dinput : _type_
        _description_
    z : _type_
        _description_
    model_parameters : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    dinput_dtheta = unfreeze(dinput_dtheta)
    for layer in dinput_dtheta['params']:
        # Matrix multiplication inform of einstein sums (only really needed for the kernel
        # calculation, but makes the code uniform)
        dinput_dtheta['params'][layer]['bias'] = jnp.einsum("ij,jk->ik", df_dinput, dinput_dtheta['params'][layer]['bias'])
        dinput_dtheta['params'][layer]['kernel'] = jnp.einsum("ij,jkl->ikl", df_dinput, dinput_dtheta['params'][layer]['kernel'])
    # dinput_dtheta, should now be called df_dtheta
    return dinput_dtheta

@jit
def g(z, z_ref, model_parameters):
    '''Calculates the inner part of the loss function.

    This function can either take individual floats for z
    and z_ref or whole numpy arrays'''
    return jnp.mean(0.5 * (z_ref - z)**2, axis = 0)

@jit
def J(z, z_ref, optimisation_parameters):
    '''Calculates the complete loss of a trajectory w.r.t. a reference trajectory'''
    return np.mean(g(z, z_ref, optimisation_parameters))

def J_residual(inputs, outputs, nn_parameters):
    def squared_error(input, output):
        pred = jitted_neural_network(nn_parameters, input)
        return (output-pred)**2
    return jnp.mean(jax.vmap(squared_error)(inputs, outputs), axis=0)[0]

def create_residuals(z_ref, t, z_dot_fmu):
    z_dot = (z_ref[1:] - z_ref[:-1])/(t[1:] - t[:-1]).reshape(-1,1)
    # v_ode = jax.vmap(lambda z_ref, t, ode_parameters: ode_res(z_ref, t, ode_parameters), in_axes=(0, 0, None))
    residual = z_dot - z_dot_fmu
    # for z_ref_value, t_value in np.dstack((z_ref[:-1], t[:-1])):
    #     residuals.append(z_dot - pointers.dx)
    return np.asarray(residual)

def f_euler(z0, t, fmu_evaluator: FMUEvaluator, model, model_parameters=None, save_derivatives=False):
    '''Applies euler to the VdP ODE by calling the fmu; returns the trajectory'''
    z = np.zeros((t.shape[0], 2))
    derivatives= np.zeros((t.shape[0], 2))
    z[0] = z0
    # Forward the initial state to the FMU
    fmu_evaluator.setup_initial_state(z0)
    times = []
    if fmu_evaluator.training:
        dfmu_dz_trajectory = []
        dfmu_dinput_trajectory = []
    # derivatives_list = []
    for i in range(len(t)-1):
        # start = time.time()
        status = fmu_evaluator.fmu.setTime(t[i])
        dt = t[i+1] - t[i]

        if fmu_evaluator.training:
            enterEventMode, terminateSimulation, dfmu_dz_at_t, dfmu_dinput_at_t = fmu_evaluator.evaluate_fmu(t[i], dt, model, model_parameters)
            z[i+1] = z[i] + dt * fmu_evaluator.pointers.dx
            if save_derivatives:
                derivatives[i] = fmu_evaluator.pointers.dx
            dfmu_dz_trajectory.append(dfmu_dz_at_t)
            dfmu_dinput_trajectory.append(dfmu_dinput_at_t)
        else:
            enterEventMode, terminateSimulation = fmu_evaluator.evaluate_fmu(t[i], dt, model, model_parameters)
            z[i+1] = z[i] + dt * fmu_evaluator.pointers.dx
            if save_derivatives:
                derivatives[i] = fmu_evaluator.pointers.dx

        if terminateSimulation:
            break

    # We get on jacobian less then we get datapoints, since we get the jacobian
    # with every derivative we calculate, and we have one datapoint already given
    # at the start
    if fmu_evaluator.training:
        enterEventMode, terminateSimulation, dfmu_dz_at_t, dfmu_dinput_at_t = fmu_evaluator.evaluate_fmu(t[i], dt, model, model_parameters)
        dfmu_dz_trajectory.append(dfmu_dz_at_t)
        dfmu_dinput_trajectory.append(dfmu_dinput_at_t)
        dfmu_dinput_trajectory = jnp.asarray(dfmu_dinput_trajectory)
        while len(dfmu_dinput_trajectory.shape) <= 2:
            dfmu_dinput_trajectory = jnp.expand_dims(dfmu_dinput_trajectory, -1)
        if save_derivatives:
            return z, np.asarray(dfmu_dz_trajectory), jnp.asarray(dfmu_dinput_trajectory), derivatives
        else:
            return z, np.asarray(dfmu_dz_trajectory), jnp.asarray(dfmu_dinput_trajectory)
    else:
        if save_derivatives:
            return z, derivatives
        else:
            return z

def adj_euler(a0, z, z_ref, t, optimisation_parameters, df_dz_trajectory):
    '''Applies forward Euler to the adjoint ODE and returns the trajectory'''
    a = np.zeros((t.shape[0], 2))
    a[0] = a0
    for i in range(len(t)-1):
        dt = t[i+1] - t[i]
        d_adj = adjoint_f(a[i], z[i], z_ref[i], t[i], optimisation_parameters, df_dz_trajectory[i]) # lang, aber die steps brauchen nicht lang?
        a[i+1] = a[i] + dt * d_adj
    return a

# For calculation of the reference solution we need the correct behaviour of the VdP
def damping(params, inputs):
    return params * (1 - inputs[0]**2) * inputs[1]

# For calculation with residuals we need a dummy augment model which just does nothing
def zero(params, inputs):
    return 0.0

def eval_fmu_NN(fmu_evaluator_NN, Tstart, Tend, params, inputs):
    output = fmu_evaluator_NN.evaluate_nn_fmu(t=Tend, inputs=inputs)
    fmu_evaluator_NN.reset_fmu(Tstart, Tend)
    return output

def optimisation_wrapper(flat_model_parameters, args):
    """This is a function wrapper for the optimisation function. It returns the
    loss and the jacobian of the loss function with respect to the optimisation
    parameters.

    Parameters
    ----------
    flat_model_parameters : _type_
        _description_
    args : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """

    start = time.time()

    t = args['t']
    z0 = args['z0']
    z_ref = args['z_ref']
    fmu_evaluator = args['fmu_evaluator']
    model = args['model']
    unravel_pytree = args['unravel_pytree']
    epoch = args['epoch']
    vectorized_df_dtheta_function = args['vectorized_df_dtheta_function']
    vectorized_df_dz_function = args['vectorized_df_dz_function']
    vectorized_dinput_dz_function = args['vectorized_dinput_dz_function']
    vectorized_dg_dtheta_function = args['vectorized_dg_dtheta_function']
    dinput_dtheta_function = args['dinput_dtheta_function']


    # Unflatten the model parameters so they fit into the neural network
    model_parameters = unravel_pytree(flat_model_parameters)

    # Calculate Forward solution and loss
    z, dfmu_dz_trajectory, dfmu_dinput_trajectory = f_euler(z0, t, fmu_evaluator, model, model_parameters) # 0.06-0.09 sec
    fmu_evaluator.reset_fmu()
    loss = J(z, z_ref, model_parameters)

    # Calculate gradients for optimisation
    dinput_dz_trajectory = vectorized_dinput_dz_function(model_parameters, z)
    df_dz_trajectory = vectorized_df_dz_function(dfmu_dz_trajectory, dinput_dz_trajectory, dfmu_dinput_trajectory)
    dinput_dtheta = dinput_dtheta_function(model_parameters, z)

    # Calculate Adjoing solution
    a0 = np.array([0, 0])
    adjoint = adj_euler(a0, np.flip(z, axis=0), np.flip(z_ref, axis=0), np.flip(t), model_parameters, np.flip(np.asarray(df_dz_trajectory), axis=0)) # 0.025-0.035
    adjoint = np.flip(adjoint, axis=0)


    # Calculate the wanted gradient with respect to the NN parameters
    df_dtheta_trajectory = vectorized_df_dtheta_function(dfmu_dinput_trajectory, dinput_dtheta)
    df_dtheta_trajectory = unfreeze(df_dtheta_trajectory)
    for layer in df_dtheta_trajectory['params']:
        # Sum the matmul result over the entire time_span to get the final gradients
        df_dtheta_trajectory['params'][layer]['bias'] = np.einsum("Ni,Nij->j", adjoint, df_dtheta_trajectory['params'][layer]['bias'])
        df_dtheta_trajectory['params'][layer]['kernel'] = np.einsum("Ni,Nijk->jk", adjoint, df_dtheta_trajectory['params'][layer]['kernel'])
    df_dtheta, _ = flatten_util.ravel_pytree(df_dtheta_trajectory)

    # This evaluates only to zeroes, but for completeness sake
    dg_dtheta_at_t = vectorized_dg_dtheta_function(z, z_ref, flat_model_parameters)
    dg_dtheta = jnp.einsum("Ni->i", dg_dtheta_at_t)

    dJ_dtheta = dg_dtheta + df_dtheta

    args['epoch'] += 1
    end = time.time()
    print(f'Epoch: {epoch}, Loss: {loss:.5f}, Time: {end-start:3.5f}')
    return loss, dJ_dtheta


def residual_wrapper(model_parameters, args):
    """_summary_

    Parameters
    ----------
    model_parameters : _type_
        _description_
    args : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """

    start = time.time()
    t = args['t']
    z0 = args['z0']
    z_ref = args['z_ref']
    z_dot_fmu = args['z_dot_fmu']
    fmu_evaluator = args['fmu_evaluator']
    model = args['model']
    unravel_pytree = args['unravel_pytree']
    epoch = args['epoch']

    model_parameters = unravel_pytree(model_parameters)

    outputs = create_residuals(z_ref, t, z_dot_fmu)[:,1]
    inputs = z_ref[:-1]

    z = f_euler(z0, t, fmu_evaluator, model, model_parameters)
    res_loss, gradient = jax.value_and_grad(J_residual, argnums=2)(inputs, outputs, model_parameters)
    true_loss = float(J(z, z_ref, model_parameters))
    flat_gradient, _ = flatten_util.ravel_pytree(gradient)
    epoch += 1
    end = time.time()
    print(f'Residuals: Epoch: {epoch}, Residual Loss: {res_loss:.10f}, Trajectory Loss: {true_loss:.10f}, Time: {end-start:3.3f}')
    args['epoch'] += 1
    return res_loss, flat_gradient

# The Neural Network structure class
class ExplicitMLP(nn.Module):
    features: Sequence[int]

    def setup(self):
        # we automatically know what to do with lists, dicts of submodules
        self.layers = [nn.Dense(feat) for feat in self.features]
        # for single submodules, we would just write:
        # self.layer1 = nn.Dense(feat1)

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
    jitted_neural_network = jax.jit(lambda params, inputs: neural_network.apply(params, inputs))
    return neural_network, jitted_neural_network, nn_parameters

if __name__ == '__main__':
    # ODE SETUP
    ####################################################################################
    Tstart = 0.0
    Tend = 20.0
    nSteps = 1001
    t = np.linspace(Tstart, Tend, nSteps)
    z0 = np.array([1.0, 0.0])
    mu = 3.0
    second_run = False
    n_inputs = 2
    n_outputs = 1

    # NEURAL NETWORK
    ####################################################################################
    layers = [15, n_outputs]

    neural_network, jitted_neural_network, nn_parameters = create_nn(layers, z0)

    # key1, key2 = random.split(random.PRNGKey(0), 2)
    # neural_network = ExplicitMLP(features=layers)
    # neural_network_parameters = neural_network.init(key2, np.zeros((1, n_inputs)))
    # jitted_neural_network = jax.jit(lambda params, inputs: neural_network.apply(params, inputs))
    flat_nn_parameters, unravel_pytree = flatten_util.ravel_pytree(nn_parameters)

    # FMU SETUP
    ####################################################################################
    fmu_filename = 'Van_der_Pol_damping_input.fmu'
    path = os.path.abspath(__file__)
    fmu_filename = '/'.join(path.split('/')[:-1]) + '/' + fmu_filename
    fmu_evaluator = FMUEvaluator(fmu_filename, Tstart, Tend)

    # REFERENCE SOLUTION
    ####################################################################################
    z_ref = f_euler(z0=z0, t=t, fmu_evaluator=fmu_evaluator, model=damping, model_parameters=mu, save_derivatives=False)
    fmu_evaluator.reset_fmu(Tstart, Tend)

    # PURE FMU MODEL WITH DERIVATIVES FOR RESIDUALS
    ####################################################################################
    _, z_dot_fmu = f_euler(z0=z0, t=t, fmu_evaluator=fmu_evaluator, model=zero, model_parameters=None, save_derivatives=True)
    z_dot_fmu = z_dot_fmu[:-1] # We create comparison derivatives with finite Differences, that means we lose one entry
    fmu_evaluator.reset_fmu(Tstart, Tend)

    # OPTIMISATION
    ####################################################################################
    vectorized_df_dtheta_function = jax.jit(jax.vmap(df_dtheta_function, in_axes=(0, 0)))

    df_dz_function = lambda dfmu_dz, dinput_dz, dfmu_dinput: dfmu_dz + dinput_dz * dfmu_dinput
    vectorized_df_dz_function = jax.jit(jax.vmap(df_dz_function, in_axes=(0,0,0)))

    dinput_dz_function = lambda p, z: jnp.array(jax.jacobian(neural_network.apply, argnums=1)(p, z))
    # dinput_dz_function = lambda p, z: jnp.array(jax.jacobian(jitted_neural_network, argnums=1)(p, z))
    vectorized_dinput_dz_function = jax.jit(jax.vmap(dinput_dz_function, in_axes=(None, 0)))

    dg_dtheta_function = lambda z, z_ref, theta: jnp.array(jax.grad(g, argnums=2)(z, z_ref, theta))
    vectorized_dg_dtheta_function = jit(jax.vmap(dg_dtheta_function, in_axes=(0, 0, None)))

    dinput_dtheta_function = lambda p, z: jax.jacobian(neural_network.apply, argnums=0)(p, z)
    # dinput_dtheta_function = lambda p, z: jax.jacobian(jitted_neural_network, argnums=0)(p, z)
    dinput_dtheta_function = jax.jit(dinput_dtheta_function)

    epoch = 0

    args_trajectory = {
        't': t,
        'z0': z0,
        'z_ref': z_ref,
        'model': jitted_neural_network,
        'fmu_evaluator': fmu_evaluator,
        'jitted_neural_network': jitted_neural_network,
        'unravel_pytree': unravel_pytree,
        'epoch': epoch,
        'vectorized_df_dtheta_function': vectorized_df_dtheta_function,
        'vectorized_df_dz_function': vectorized_df_dz_function,
        'vectorized_dinput_dz_function': vectorized_dinput_dz_function,
        'vectorized_dg_dtheta_function': vectorized_dg_dtheta_function,
        'dinput_dtheta_function': dinput_dtheta_function,
    }

    args_residual = {
        't': t,
        'z0': z0,
        'z_ref': z_ref,
        'model': jitted_neural_network,
        'fmu_evaluator': fmu_evaluator,
        'jitted_neural_network': jitted_neural_network,
        'unravel_pytree': unravel_pytree,
        'epoch': epoch,
        'z_dot_fmu': z_dot_fmu
    }



    # Optimise the mu value via scipy
    # Optimisers: CG, BFGS, Newton-CG, L-BFGS-B, TNC, SLSQP, dogleg, trust-ncg, trust-krylov, trust-exact and trust-constr
    fmu_evaluator.training = False
    res = minimize(residual_wrapper, flat_nn_parameters, method='BFGS', jac=True, args=args_residual, tol=1e-8)
    flat_nn_parameters = res.x
    neural_network_parameters = unravel_pytree(flat_nn_parameters)
    fmu_evaluator.reset_fmu(Tstart, Tend)
    fmu_evaluator.training = False
    z = f_euler(z0, t, fmu_evaluator, model=jitted_neural_network, model_parameters=neural_network_parameters)
    plot_path = get_file_path(path)
    plot_results(t, z, z_ref, plot_path+f'_mu_{mu}_residual')

    fmu_evaluator.training = True
    res = minimize(optimisation_wrapper, flat_nn_parameters, method='BFGS', jac=True, args=args_trajectory, tol=1e-8)
    flat_nn_parameters = res.x
    neural_network_parameters = unravel_pytree(flat_nn_parameters)
    fmu_evaluator.reset_fmu(Tstart, Tend)
    fmu_evaluator.training = False
    z = f_euler(z0, t, fmu_evaluator, model=jitted_neural_network, model_parameters=neural_network_parameters)
    plot_path = get_file_path(path)
    plot_results(t, z, z_ref, plot_path+f'_mu_{mu}_trajectory')

    fmu_evaluator.reset_fmu(Tstart, Tend)

    # SECOND REFERENCE AND OPTIMISATION RUN WITH MU=8.53; Is the training better if we
    # pretrain with a another mu value?
    if second_run:

        fmu_evaluator.training = False
        mu = 8.53
        z_ref = f_euler(z0=z0, t=t, fmu_evaluator=fmu_evaluator, model=damping, model_parameters=mu)
        # Reset and reinitialize the fmu for the next run after the reference run
        fmu_evaluator.reset_fmu(Tstart, Tend)
        epoch = 0

        args = [t,
                z0,
                z_ref,
                fmu_evaluator,
                jitted_neural_network,
                vectorized_df_dtheta_function,
                vectorized_df_dz_function,
                vectorized_dinput_dz_function,
                vectorized_dg_dtheta_function,
                dinput_dtheta_function,
                unravel_pytree,
                epoch,
                None]

        fmu_evaluator.training = True

        flat_nn_parameters, unravel_pytree = flatten_util.ravel_pytree(neural_network_parameters)

        # Optimise the mu value via scipy
        # Optimisers CG, BFGS, Newton-CG, L-BFGS-B, TNC, SLSQP, dogleg, trust-ncg, trust-krylov, trust-exact and trust-constr
        # BFGS seems not to terminate as quickly even if the loss change is small
        res = minimize(optimisation_wrapper, flat_nn_parameters, method='BFGS', jac=True, args=args, tol=1e-8)
        print(res)
        # The values we optimized for are inside the result variable
        neural_network_parameters = res.x
        neural_network_parameters = unravel_pytree(neural_network_parameters)
        # Reset the FMU again for a final run for plotting purposes
        fmu_evaluator.reset_fmu(Tstart, Tend)

    # PLOT SOLUTION WITH REFERENCE
    ####################################################################################
    fmu_evaluator.training = False

    z = f_euler(z0, t, fmu_evaluator, model=jitted_neural_network, model_parameters=neural_network_parameters)
    fmu_evaluator.reset_fmu(Tstart, Tend)
    print(z[-10:])

    path = os.path.abspath(__file__)
    plot_path = get_file_path(path)
    plot_results(t, z, z_ref, plot_path+f'_mu_{mu}')

    Tstart_NN = 0.0
    Tend_NN = 1.0
    nSteps_NN = 2
    t_NN = np.linspace(Tstart_NN, Tend_NN, nSteps_NN)
    z0 = np.array([1.0, 0.0])

    modelName = "NN"
    directory = '/'.join(path.split('/')[:-1])
    path  = os.path.join(directory, f"{modelName}", "build", f"{modelName}.fmu")
    targetDirPath = os.path.join(directory, f"{modelName}")

    create_NN_FMU(targetDirPath, modelName, neural_network_parameters, n_inputs, n_outputs)

    fmu_evaluator_NN = FMUEvaluator(path, Tstart_NN, Tend_NN)

    nn_fmu_eval_function = partial(eval_fmu_NN, fmu_evaluator_NN=fmu_evaluator_NN, Tstart=Tstart_NN, Tend=Tend_NN)

    print(f"Python NN prediction for [1.0, 0.0]: {jitted_neural_network(neural_network_parameters, jnp.array([1.0, 0.0]))}")
    print(f"FMU NN prediction for [1.0, 0.0]: {nn_fmu_eval_function(params=None, inputs=[1.0, 0.0])}")

    z_fmu = f_euler(z0, t, fmu_evaluator, model=nn_fmu_eval_function, model_parameters=None)
    plot_results(t, z_fmu, z_ref, plot_path+f'_mu_{mu}_fmu_load')
    fmu_evaluator.reset_fmu(Tstart, Tend)

    print(z_fmu[-10:])
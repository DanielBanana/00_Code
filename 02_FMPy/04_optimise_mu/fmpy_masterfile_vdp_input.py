
# For interaction with the FMU
import fmpy
from fmpy import read_model_description, extract
from fmpy.fmi2 import _FMU2 as FMU2
from fmpy.util import plot_result, download_test_file
import ctypes
from types import SimpleNamespace

# For automatic differentiaton
import jax
from jax import jit, numpy as jnp

# For optimisation
from scipy.optimize import minimize

# General
import numpy as np
import shutil
from matplotlib import pyplot as plt
import os
import sys

# To use the plot_results file we need to add the uppermost folder to the PYTHONPATH
# Only Works if file gets called from 00_Code
sys.path.insert(0, os.getcwd())
from plot_results import plot_results, get_plot_path
from jax.config import config
config.update("jax_debug_nans", False)
config.update("jax_enable_x64", True)

# def settable_in_instantiated(variable):
#     return variable.causality == 'input' \
#            or variable.variability != 'constant' and variable.initial in {'approx', 'exact'}

def prepare_fmu(fmu_filename, Tstart, Tend):
    model_description = read_model_description(fmu_filename)
    # extract the FMU
    unzipdir = extract(fmu_filename)
    fmu = fmpy.fmi2.FMU2Model(guid=model_description.guid,
                    unzipDirectory=unzipdir,
                    modelIdentifier=model_description.modelExchange.modelIdentifier,
                    instanceName='instance1')
    # instantiate
    fmu.instantiate()

    # set variable start values (of "ScalarVariable / <type> / start")
    pass

    # initialize
    # determine continous and discrete states
    fmu.setupExperiment(startTime=Tstart, stopTime=Tend)
    fmu.enterInitializationMode()

    # set the input start values at time = Tstart
    pass

    fmu.exitInitializationMode()

    nx = model_description.numberOfContinuousStates
    nz = model_description.numberOfEventIndicators
    initialEventMode = False
    enterEventMode = False
    timeEvent = False
    stateEvent = False
    previous_z = np.zeros(nz)

    # retrieve initial state x and
    # nominal values of x (if absolute tolerance is needed)
    # pointers to exchange state and derivative vectors with FMU
    pointers = SimpleNamespace(
        x=np.zeros(nx),
        dx=np.zeros(nx),
        z=np.zeros(nz),
    )
    pointers._px = pointers.x.ctypes.data_as(
        ctypes.POINTER(ctypes.c_double)
    )
    pointers._pdx = pointers.dx.ctypes.data_as(
        ctypes.POINTER(ctypes.c_double)
    )
    pointers._pz = pointers.z.ctypes.data_as(
        ctypes.POINTER(ctypes.c_double)
    )
    status = fmu.getContinuousStates(pointers._px, pointers.x.size)

    # collect the value references
    vrs = {}
    for variable in model_description.modelVariables:
        vrs[variable.name] = variable.valueReference

    # get the value references for the variables we want to get/set
    vr_states   = [vrs['u'], vrs['v']]
    vr_derivatives = [vrs['der(u)'], vrs['der(v)']]
    vr_input = [vrs['mu']]

    fmu.enterContinuousTimeMode()

    return fmu, model_description, pointers, vr_states, vr_derivatives, vr_input

def reset_fmu(fmu, model_description, Tstart, Tend):
    fmu.reset()
    nx = model_description.numberOfContinuousStates
    nz = model_description.numberOfEventIndicators
    # initialize
    # determine continous and discrete states
    fmu.setupExperiment(startTime=Tstart, stopTime=Tend)
    fmu.enterInitializationMode()

    # set the input start values at time = Tstart
    pass

    fmu.exitInitializationMode()

    # retrieve initial state x and
    # nominal values of x (if absolute tolerance is needed)
    # pointers to exchange state and derivative vectors with FMU
    pointers = SimpleNamespace(
        x=np.zeros(nx),
        dx=np.zeros(nx),
        z=np.zeros(nz),
    )
    pointers._px = pointers.x.ctypes.data_as(
        ctypes.POINTER(ctypes.c_double)
    )
    pointers._pdx = pointers.dx.ctypes.data_as(
        ctypes.POINTER(ctypes.c_double)
    )
    pointers._pz = pointers.z.ctypes.data_as(
        ctypes.POINTER(ctypes.c_double)
    )

    fmu.enterContinuousTimeMode()

    return fmu, pointers

def f(fmu, pointers, number_of_states, vr_derivatives, vr_states, vr_input):
    status = fmu.getDerivatives(pointers._pdx, pointers.dx.size)
    df_dz_at_t = df_dz_function(fmu, number_of_states, vr_derivatives, vr_states)
    df_dphi_at_t = df_dphi_function(fmu, vr_derivatives, vr_input)
    return pointers.dx, df_dz_at_t, df_dphi_at_t

@jit
def adjoint_f(adj, z, z_ref, t, optimisation_parameters, df_dz_at_t):
    '''Calculates the right hand side of the adjoint system.'''
    dg_dz = jax.grad(g, argnums=0)(z, z_ref, optimisation_parameters)
    d_adj = - df_dz_at_t.T @ adj - dg_dz
    return d_adj

# Returns a Jacobian
def df_dz_function(fmu, number_of_states, vr_derivatives, vr_states):
    current_df_dz = np.zeros((2,2))
    for j in range(number_of_states):
            current_df_dz[:, j] = np.array(fmu.getDirectionalDerivative(vr_derivatives, [vr_states[j]], [1.0]))
    return current_df_dz

# Returns a Jacobian
def df_dphi_function(fmu, vr_derivatives, vr_input):
    return fmu.getDirectionalDerivative(vr_derivatives, vr_input, [1.0])

def g(z, z_ref, ode_parameters):
    '''Calculates the inner part of the loss function.

    This function can either take individual floats for z
    and z_ref or whole numpy arrays'''
    return jnp.sum(0.5 * (z_ref - z)**2, axis = 0)

def J(z, z_ref, optimisation_parameters):
    '''Calculates the complete loss of a trajectory w.r.t. a reference trajectory'''
    return np.sum(g(z, z_ref, optimisation_parameters))

def f_euler(z0, t, fmu, pointers, number_of_states, vr_derivatives, vr_states, vr_input):
    '''Applies euler to the VdP ODE by calling the fmu; returns the trajectory'''
    z = np.zeros((t.shape[0], 2))
    z[0] = z0

    # Forward the initial state to the FMU
    fmu.setReal(vr_states, z0)
    fmu.getContinuousStates(pointers._px, pointers.x.size)

    df_dz_trajectory = []
    df_dphi_trajectory = []
    for i in range(len(t)-1):
        status = fmu.setTime(t[i])
        dt = t[i+1] - t[i]

        derivatives, df_dz_at_t, df_dphi_at_t = f(fmu, pointers, number_of_states, vr_derivatives, vr_states, vr_input)
        z[i+1] = z[i] + dt * derivatives
        df_dz_trajectory.append(df_dz_at_t)
        df_dphi_trajectory.append(df_dphi_at_t)
        pointers.x += dt * pointers.dx

        status = fmu.setContinuousStates(pointers._px, pointers.x.size)

        # get event indicators at t = time
        status = fmu.getEventIndicators(pointers._pz, pointers.z.size)


                # inform the model about an accepted step
        enterEventMode, terminateSimulation = fmu.completedIntegratorStep()

        # get continuous output
        # fmu.getReal([vr_outputs])

        if terminateSimulation:
            break
    return z, np.asarray(df_dz_trajectory), np.asarray(df_dphi_trajectory)

def adj_euler(a0, z, z_ref, t, optimisation_parameters, df_dz_trajectory):
    '''Applies forward Euler to the adjoint ODE and returns the trajectory'''
    a = np.zeros((t.shape[0], 2))
    a[0] = a0
    for i in range(len(t)-1):
        dt = t[i+1] - t[i]
        #(adj, z, z_ref, t, optimisation_parameters, fmu_parameters):
        a[i+1] = a[i] + dt * adjoint_f(a[i], z[i], z_ref[i], t[i], optimisation_parameters, df_dz_trajectory[i])
    return a

# Vectorize the  jacobian dg_dphi for all time points
dg_dphi_function = lambda z, z_ref, phi: jnp.array(jax.grad(g, argnums=2)(z, z_ref, phi))
vectorized_dg_dphi_function = jit(jax.vmap(dg_dphi_function, in_axes=(0, 0, None)))

def optimisation_wrapper(optimisation_parameters, args):
    '''This is a function wrapper for the optimisation function. It returns the
    loss and the jacobian'''
    print(f'mu: {optimisation_parameters}')
    t = args[0]
    z0 = args[1]
    z_ref = args[2]
    fmu = args[3]
    pointers = args[4]
    number_of_states = args[5]
    vr_derivatives = args[6]
    vr_states = args[7]
    vr_input = args[8]

    fmu.setReal(vr_input, [optimisation_parameters[0]])

    z, df_dz_trajectory, df_dphi_trajectory = f_euler(z0, t, fmu, pointers, number_of_states, vr_derivatives, vr_states, vr_input)
    loss = J(z, z_ref, optimisation_parameters)

    a0 = np.array([0, 0])
    adjoint = adj_euler(a0, np.flip(z, axis=0), np.flip(z_ref, axis=0), np.flip(t), optimisation_parameters, np.flip(df_dz_trajectory, axis=0))
    adjoint = np.flip(adjoint, axis=0)

    if len(df_dphi_trajectory.shape) == 2:
        df_dphi_trajectory = jnp.expand_dims(df_dphi_trajectory, 2)
        df_dphi = float(np.einsum("Ni,Nij->j", adjoint[1:], df_dphi_trajectory))
    else:
        df_dphi = jnp.einsum("Ni,Nij->j", adjoint[1:], df_dphi_trajectory)

    # This evaluates only to zeroes, but for completeness sake
    dg_dphi_at_t = vectorized_dg_dphi_function(z, z_ref, optimisation_parameters)
    dg_dphi = jnp.einsum("Ni->i", dg_dphi_at_t)

    dJ_dphi = dg_dphi + df_dphi

    reset_fmu(fmu, model_description, Tstart, Tend)

    # print(f'Loss: {loss}; Mu: {optimisation_parameters}; gradient: {dJ_dphi}')
    # print(optimisation_parameters)
    print(f'Jacobian/Gradient: {dJ_dphi}')
    return loss, dJ_dphi


if __name__ == '__main__':
    # mu = float(input('Set mu value: '))
    Tstart = 0.0
    Tend = 15.0
    nSteps = 601
    t = np.linspace(Tstart, Tend, nSteps)
    z0 = np.array([1.0, 0.0])
    optimisation_parameters_ref = np.asarray([5.0])
    optimisation_parameters = np.asarray([1.0])

    fmu_filename = 'Van_der_Pol_input.fmu'

    path = os.path.abspath(__file__)
    fmu_filename = '/'.join(path.split('/')[:-1]) + '/' + fmu_filename

    # Readout the model description and load the fmu into python
    fmu, model_description, pointers, vr_states, vr_derivatives, vr_input = prepare_fmu(fmu_filename,  Tstart, Tend)
    number_of_states = model_description.numberOfContinuousStates

    # Set the reference value for mu inside the FMU
    fmu.setReal(vr_input, [optimisation_parameters_ref[0]])

    # Calculate the reference solution via the fmu and the reference mu value
    z_ref, _, __ = f_euler(z0, t, fmu, pointers, number_of_states, vr_derivatives, vr_states, vr_input)

    fig = plt.figure()
    ax1, ax2 = fig.subplots(2, 1)
    ax1.plot(t, z_ref[:,0])
    ax1.plot(t, z_ref[:,1])
    ax1.set_title('Reference')

    # Pack all values needed for the opimisation into one array to give
    # to the minimize function
    args = [t, z0, z_ref, fmu, pointers, number_of_states, vr_derivatives, vr_states, vr_input]

    # Reset and reinitialize the fmu for the next run after the reference run
    fmu, pointers = reset_fmu(fmu, model_description, Tstart, Tend)

    # Optimise the mu value via scipy
    # Optimisers CG, BFGS, Newton-CG, L-BFGS-B, TNC, SLSQP, dogleg, trust-ncg, trust-krylov, trust-exact and trust-constr
    res = minimize(optimisation_wrapper, optimisation_parameters, method='BFGS', jac=True, args=args)
    print(res)

    # The values we optimized for are inside the result variable
    optimisation_parameters = res.x

    # Reset the FMU again for a final run for plotting purposes
    reset_fmu(fmu, model_description, Tstart, Tend)
    fmu.setReal(vr_input, [optimisation_parameters[0]])
    z, _, __ = f_euler(z0, t, fmu, pointers, number_of_states, vr_derivatives, vr_states, vr_input)

    ax2.plot(t, z[:,0])
    ax2.plot(t, z[:,1])
    ax2.set_title('Solution')
    plt.show()
    fig.savefig('fmu_adjoint.png')

    path = os.path.abspath(__file__)
    plot_path = get_plot_path(path)
    plot_results(t, z, z_ref, plot_path)

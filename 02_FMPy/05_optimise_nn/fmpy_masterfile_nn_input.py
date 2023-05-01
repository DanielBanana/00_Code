
# For interaction with the FMU
import fmpy
from fmpy import read_model_description, extract
from fmpy.fmi2 import _FMU2 as FMU2
import ctypes
from types import SimpleNamespace

# For automatic differentiaton
import jax
from jax import random, jit, flatten_util, numpy as jnp
from flax import linen as nn
from flax.core import unfreeze
from typing import Sequence, List

# For optimisation
from scipy.optimize import minimize

# General
import numpy as np
from matplotlib import pyplot as plt
import os
import sys
import time

# To use the plot_results file we need to add the uppermost folder to the PYTHONPATH
# Only Works if file gets called from 00_Code
sys.path.insert(0, os.getcwd())
from plot_results import plot_results, get_plot_path
from jax.config import config
config.update("jax_debug_nans", False)
config.update("jax_enable_x64", True)

class FMUEvaluator:
    """Class which manages the evaluation of the FMU with python; stores important
    variables like the fmu instance, variable references. Also manages the evaluation
    with a machine learning model: One can switch between training and evaluation mode.
    In training mode relevant derivatives and jacobians get caluclated, not so in evaluation
    mode.
    """
    fmu: fmpy.fmi2.FMU2Model
    fmu_filename: str
    model_description: fmpy.model_description.ModelDescription
    vr_states: List
    vr_derivatives: List
    vr_inputs: List
    vr_outputs: List
    n_states: int
    n_events: int
    Tstart: float
    Tend: float
    training: bool

    def __init__(self, fmu_filename, Tstart, Tend):
        self.fmu_filename = fmu_filename
        self.Tstart = Tstart
        self.Tend = Tend

        self.model_description = read_model_description(fmu_filename)
        # extract the FMU
        unzipdir = extract(fmu_filename)
        self.fmu = fmpy.fmi2.FMU2Model(guid=self.model_description.guid,
                        unzipDirectory=unzipdir,
                        modelIdentifier=self.model_description.modelExchange.modelIdentifier,
                        instanceName='instance1')
        # instantiate
        self.fmu.instantiate()

        # set variable start values (of "ScalarVariable / <type> / start")
        pass

        # initialize
        # determine continous and discrete states
        self.fmu.setupExperiment(startTime=Tstart, stopTime=Tend)
        self.fmu.enterInitializationMode()

        # set the input start values at time = Tstart
        pass

        self.fmu.exitInitializationMode()

        self.n_states = self.model_description.numberOfContinuousStates
        self.n_events = self.model_description.numberOfEventIndicators
        initialEventMode = False
        enterEventMode = False
        timeEvent = False
        stateEvent = False
        previous_z = np.zeros(self.n_events)

        # retrieve initial state x and
        # nominal values of x (if absolute tolerance is needed)
        # pointers to exchange state and derivative vectors with FMU
        pointers = SimpleNamespace(
            x=np.zeros(self.n_states),
            dx=np.zeros(self.n_states),
            z=np.zeros(self.n_events),
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
        status = self.fmu.getContinuousStates(pointers._px, pointers.x.size)
        self.pointers = pointers

        # collect the value references
        vrs = {}
        for variable in self.model_description.modelVariables:
            vrs[variable.name] = variable.valueReference

        # get the value references for the variables we want to get/set
        self.vr_states   = [vrs['u'], vrs['v']]
        self.vr_derivatives = [vrs['der(u)'], vrs['der(v)']]
        self.vr_input = [vrs['damping']]

        self.fmu.enterContinuousTimeMode()

        self.training = False

    def reset_fmu(self, Tstart=None, Tend=None):
        """Reset the FMU, such that a new run can be started. New start and end time
        can be given.

        Parameters
        ----------
        Tstart : float, optional
            Start time for the FMU, by default None
        Tend : float, optional
            Start time for the FMU, by default None
        """

        if Tstart is None:
            Tstart = self.Tstart

        if Tend is None:
            Tend = self.Tend

        self.fmu.reset()
        self.fmu.setupExperiment(startTime=Tstart, stopTime=Tend)
        self.fmu.enterInitializationMode()
        self.Tstart = Tstart
        self.Tend = Tend

        # set the input start values at time = Tstart
        pass

        self.fmu.exitInitializationMode()

        # retrieve initial state x and
        # nominal values of x (if absolute tolerance is needed)
        # pointers to exchange state and derivative vectors with FMU
        pointers = SimpleNamespace(
            x=np.zeros(self.n_states),
            dx=np.zeros(self.n_states),
            z=np.zeros(self.n_events),
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
        self.pointers = pointers

        self.fmu.enterContinuousTimeMode()

    def evaluate_fmu(self, t, dt, augment_model_function, augment_model_args):
        """Evaluation function of the FMU. Uses the internally stored FMU and a
        augment model function to calculate the derivatives for the
        calculation of the new state. Arguments for the function
        must be given to be directly insertable into the augment model function

        Parameters
        ----------
        t : float
            The current time
        dt : _type_
            The current time step
        augment_model_function : function
            Function which should be evaluated for the contribution of the machine learning
            model. Must take its own arguments as first input and the current state vector
            as second input
        augment_model_args :
            Arguments like weights and biases for the machine learning function

        Returns
        -------
        _type_
            Derivatives for the new state, Flag whether event mode has been entered,
            Flag whether the simulation needs to be terminated; In Training mode:
            dfmu_dz, dfmu_dinput
        """
        status = self.fmu.setTime(t)

        if self.training:
            control = augment_model_function(augment_model_args, self.pointers.x)
            self.fmu.setReal(self.vr_input, [control])
            status = self.fmu.getDerivatives(self.pointers._pdx, self.pointers.dx.size)
            dfmu_dz_at_t = self.dfmu_dz_function()
            dfmu_dinput_at_t = self.dfmu_dinput_function()
        else:
            control = augment_model_function(augment_model_args, self.pointers.x)
            self.fmu.setReal(self.vr_input, [control])
            status = self.fmu.getDerivatives(self.pointers._pdx, self.pointers.dx.size)

        # z[i+1] = z[i] + dt * derivatives

        self.pointers.x += dt * self.pointers.dx

        status = self.fmu.setContinuousStates(self.pointers._px, self.pointers.x.size)

        # get event indicators at t = time
        status = self.fmu.getEventIndicators(self.pointers._pz, self.pointers.z.size)

        # inform the model about an accepted step
        enterEventMode, terminateSimulation = self.fmu.completedIntegratorStep()

        # get continuous output
        # fmu.getReal([vr_outputs])

        # If we are in Training mode return the derivatives for the next step,
        # FMU information and Optimisation jacobians; otherwise leave out jacobians
        if self.training:
            return self.pointers.dx, enterEventMode, terminateSimulation, dfmu_dz_at_t, dfmu_dinput_at_t
        else:
            return self.pointers.dx, enterEventMode, terminateSimulation

    def dfmu_dz_function(self):
        """Calculate the jacobian of the fmu function w.r.t. the state variables

        Parameters
        ----------
        fmu : fmpy.fmi2.FMU2Model
            The python object which controls the FMU
        number_of_states : int
            How many states the FMU equation has (often times 2)
        vr_derivatives : List of int
            The variable reference numbers of the derivative variables. Each variable
            in the FMU has a indexing number.
        vr_states : List of int
            The variable reference numbers of the state variables. Each variable
            in the FMU has a indexing number.

        Returns
        -------
        _type_
            The jacobian as a numpy array
        """
        dfmu_dz = np.zeros((2,2))
        for j in range(self.n_states):
                dfmu_dz[:, j] = np.array(self.fmu.getDirectionalDerivative(self.vr_derivatives, [self.vr_states[j]], [1.0]))
        return dfmu_dz

    def dfmu_dinput_function(self):
        """Calculate the jacobian of the fmu function w.r.t. the input/control variables

        Parameters
        ----------
        fmu : fmpy.fmi2.FMU2Model
            The python object which controls the FMU
        vr_derivatives : List of int
            The variable reference numbers of the derivative variables. Each variable
            in the FMU has a indexing number.
        vr_input : List of int
            The variable reference numbers of the input variables. Each variable
            in the FMU has a indexing number.

        Returns
        -------
        _type_
            The jacobian as a numpy array
        """
        return self.fmu.getDirectionalDerivative(self.vr_derivatives, self.vr_input, [1.0])

    def setup_initial_state(self, z0):
        """Before starting the iteration of the ODE solver set the inital state in the
        FMU and load the pointers with the correct values

        Parameters
        ----------
        z0 : _type_
            _description_
        """
        self.fmu.setReal(self.vr_states, z0)
        self.fmu.getContinuousStates(self.pointers._px, self.pointers.x.size)

# Not needed anymore
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
    vr_input = [vrs['damping']]

    fmu.enterContinuousTimeMode()

    return fmu, model_description, pointers, vr_states, vr_derivatives, vr_input

# Not needed anymroe
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

# Not needed anymore
def f(fmu, mu, pointers, vr_input):
    fmu.setReal(vr_input, [damping(mu, pointers.x)])
    status = fmu.getDerivatives(pointers._pdx, pointers.dx.size)
    return pointers.dx

# Not needed anymore
def hybrid_f(fmu_evaluator, model, model_parameters):
    control = model(model_parameters, fmu_evaluator.pointers.x)
    fmu_evaluator.fmu.setReal(fmu_evaluator.vr_input, [control])
    status = fmu_evaluator.fmu.getDerivatives(fmu_evaluator.pointers._pdx, fmu_evaluator.pointers.dx.size)
    df_dz_at_t = dfmu_dz_function(fmu_evaluator.fmu, fmu_evaluator.number_of_states, fmu_evaluator.vr_derivatives, fmu_evaluator.vr_states)
    df_dinput_at_t = dfmu_dinput_function(fmu_evaluator.fmu, fmu_evaluator.vr_derivatives, fmu_evaluator.vr_input)
    return fmu_evaluator.pointers.dx, df_dz_at_t, df_dinput_at_t

@jit
def adjoint_f(adj, z, z_ref, t, optimisation_parameters, df_dz_at_t):
    '''Calculates the right hand side of the adjoint system.'''
    dg_dz = jax.grad(g, argnums=0)(z, z_ref, optimisation_parameters)
    d_adj = - df_dz_at_t.T @ adj - dg_dz
    return d_adj

# Not needed anymore
def dfmu_dz_function(fmu, number_of_states, vr_derivatives, vr_states):
    """Calculate the jacobian of the fmu function w.r.t. the state variables

    Parameters
    ----------
    fmu : fmpy.fmi2.FMU2Model
        The python object which controls the FMU
    number_of_states : int
        How many states the FMU equation has (often times 2)
    vr_derivatives : List of int
        The variable reference numbers of the derivative variables. Each variable
        in the FMU has a indexing number.
    vr_states : List of int
        The variable reference numbers of the state variables. Each variable
        in the FMU has a indexing number.

    Returns
    -------
    _type_
        The jacobian as a numpy array
    """
    current_df_dz = np.zeros((2,2))
    for j in range(number_of_states):
            current_df_dz[:, j] = np.array(fmu.getDirectionalDerivative(vr_derivatives, [vr_states[j]], [1.0]))
    return current_df_dz

@jit
def df_dz_function(dfmu_dz, dinput_dz, dfmu_dinput):
    return jnp.array(dfmu_dz + dinput_dz * dfmu_dinput)

# Not needed anymore
def dfmu_dinput_function(fmu, vr_derivatives, vr_input):
    """Calculate the jacobian of the fmu function w.r.t. the input/control variables

    Parameters
    ----------
    fmu : fmpy.fmi2.FMU2Model
        The python object which controls the FMU
    vr_derivatives : List of int
        The variable reference numbers of the derivative variables. Each variable
        in the FMU has a indexing number.
    vr_input : List of int
        The variable reference numbers of the input variables. Each variable
        in the FMU has a indexing number.

    Returns
    -------
    _type_
        The jacobian as a numpy array
    """
    return fmu.getDirectionalDerivative(vr_derivatives, vr_input, [1.0])

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

def g(z, z_ref, model_parameters):
    '''Calculates the inner part of the loss function.

    This function can either take individual floats for z
    and z_ref or whole numpy arrays'''
    return jnp.sum(0.5 * (z_ref - z)**2, axis = 0)

def J(z, z_ref, optimisation_parameters):
    '''Calculates the complete loss of a trajectory w.r.t. a reference trajectory'''
    return np.sum(g(z, z_ref, optimisation_parameters))

def f_euler(z0, t, fmu_evaluator: FMUEvaluator, model, model_parameters=None):
    '''Applies euler to the VdP ODE by calling the fmu; returns the trajectory'''
    z = np.zeros((t.shape[0], 2))
    z[0] = z0
    # Forward the initial state to the FMU
    fmu_evaluator.setup_initial_state(z0)

    if fmu_evaluator.training:
        dfmu_dz_trajectory = []
        dfmu_dinput_trajectory = []
    for i in range(len(t)-1):
        status = fmu_evaluator.fmu.setTime(t[i])
        dt = t[i+1] - t[i]

        if fmu_evaluator.training:
            derivatives, enterEventMode, terminateSimulation, dfmu_dz_at_t, dfmu_dinput_at_t = fmu_evaluator.evaluate_fmu(t[i], dt, model, model_parameters)
            dfmu_dz_trajectory.append(dfmu_dz_at_t)
            dfmu_dinput_trajectory.append(dfmu_dinput_at_t)
        else:
            derivatives, enterEventMode, terminateSimulation = fmu_evaluator.evaluate_fmu(t[i], dt, model, model_parameters)

        z[i+1] = z[i] + dt * derivatives

        if terminateSimulation:
            break

    # We get on jacobian less then we get datapoints, since we get the jacobian
    # with every derivative we calculate, and we have one datapoint already given
    # at the start
    if fmu_evaluator.training:
        derivatives, enterEventMode, terminateSimulation, dfmu_dz_at_t, dfmu_dinput_at_t = fmu_evaluator.evaluate_fmu(t[i], dt, model, model_parameters)
        dfmu_dz_trajectory.append(dfmu_dz_at_t)
        dfmu_dinput_trajectory.append(dfmu_dinput_at_t)
        dfmu_dinput_trajectory = jnp.asarray(dfmu_dinput_trajectory)
        while len(dfmu_dinput_trajectory.shape) <= 2:
            dfmu_dinput_trajectory = jnp.expand_dims(dfmu_dinput_trajectory, -1)
        return z, np.asarray(dfmu_dz_trajectory), jnp.asarray(dfmu_dinput_trajectory)
    else:
        return z

def adj_euler(a0, z, z_ref, t, optimisation_parameters, df_dz_trajectory):
    '''Applies forward Euler to the adjoint ODE and returns the trajectory'''
    a = np.zeros((t.shape[0], 2))
    a[0] = a0
    for i in range(len(t)-1):
        dt = t[i+1] - t[i]
        d_adj =  adjoint_f(a[i], z[i], z_ref[i], t[i], optimisation_parameters, df_dz_trajectory[i])
        a[i+1] = a[i] + dt * d_adj
    return a

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

# For calculation of the reference solution we need the correct behaviour of the VdP
def damping(mu, inputs):
    return mu * (1 - inputs[0]**2) * inputs[1]

def optimisation_wrapper(model_parameters, args):
    '''This is a function wrapper for the optimisation function. It returns the
    loss and the jacobian of the loss function with respect to the optimisation parameters'''
    # print(f'mu: {optimisation_parameters}')
    t = args[0]
    z0 = args[1]
    z_ref = args[2]
    fmu_evaluator = args[3]
    model = args[4]
    vectorized_df_dtheta_function = args[5]
    vectorized_df_dz_function = args[6]
    vectorized_dinput_dz_function = args[7]
    vectorized_dg_dtheta_function = args[8]
    dinput_dtheta_function = args[9]
    unravel_pytree = args[10]
    epoch = args[11]


    # start = time.time()
    model_parameters = unravel_pytree(model_parameters)

    z, dfmu_dz_trajectory, dfmu_dinput_trajectory = f_euler(z0, t, fmu_evaluator, model, model_parameters)
    loss = J(z, z_ref, model_parameters)

    # Calculating the full derivative df_dz:
    # partial_fmu_partial_z + dinput_dz*partial_fmu_aprtial_u
    start = time.time()
    dinput_dz_trajectory = vectorized_dinput_dz_function(model_parameters, z)
    cp_di_dz = time.time()
    df_dz_trajectory = vectorized_df_dz_function(dfmu_dz_trajectory, dinput_dz_trajectory, dfmu_dinput_trajectory)
    cp_df_dz = time.time()
    # time_di_dz = cp_di_dz-start # insignificant time e-5
    # time_df_dz = cp_df_dz - cp_di_dz # insignificant time e-5
    # print(f'dinput_dz time: {time_di_dz}')
    # print(f'df_dz time: {time_df_dz}')

    a0 = np.array([0, 0])
    adjoint = adj_euler(a0, np.flip(z, axis=0), np.flip(z_ref, axis=0), np.flip(t), model_parameters, np.flip(df_dz_trajectory, axis=0))
    adjoint = np.flip(adjoint, axis=0)
    cp_adjoint = time.time()
    time_adjoint = cp_adjoint - cp_df_dz #0.37 (Main time sink)
    # print(f'Adjoint time: {time_adjoint}')

    dinput_dtheta = dinput_dtheta_function(model_parameters, z)

    df_dtheta_trajectory = vectorized_df_dtheta_function(dfmu_dinput_trajectory, dinput_dtheta)

    df_dtheta_trajectory = unfreeze(df_dtheta_trajectory)

    for layer in df_dtheta_trajectory['params']:
        # Sum the matmul result over the entire time_span to get the final gradients
        df_dtheta_trajectory['params'][layer]['bias'] = np.einsum("Ni,Nij->j", adjoint, df_dtheta_trajectory['params'][layer]['bias'])
        df_dtheta_trajectory['params'][layer]['kernel'] = np.einsum("Ni,Nijk->jk", adjoint, df_dtheta_trajectory['params'][layer]['kernel'])

    df_dtheta = df_dtheta_trajectory

    # This evaluates only to zeroes, but for completeness sake
        # dg_dtheta_at_t = vectorized_dg_dtheta_function(z, z_ref, model_parameters)
        # dg_dtheta = jnp.einsum("Ni->i", dg_dtheta_at_t)
        # dJ_dtheta = dg_dtheta + df_dtheta

    dJ_dtheta = df_dtheta

    flat_dJ_dtheta, _ = flatten_util.ravel_pytree(dJ_dtheta)

    fmu_evaluator.reset_fmu()

    end = time.time()
    time_opt_step = end-start
    # print(f'Time completet step: {time_opt_step}')
    print(f'Epoch: {epoch}, Loss: {loss:.5f}')
    epoch += 1
    args[11] = epoch
    return loss, flat_dJ_dtheta


if __name__ == '__main__':
    # ODE SETUP
    ####################################################################################
    Tstart = 0.0
    Tend = 10.0
    nSteps = 1001
    t = np.linspace(Tstart, Tend, nSteps)
    z0 = np.array([1.0, 0.0])
    mu = 5.0

    # NEURAL NETWORK
    ####################################################################################
    layers = [10, 10, 1]
    key1, key2 = random.split(random.PRNGKey(0), 2)
    neural_network = ExplicitMLP(features=layers)
    # Input size is guess from input during init
    neural_network_parameters = neural_network.init(key2, np.zeros((1, 2)))
    jitted_neural_network = jax.jit(lambda p, x: neural_network.apply(p, x))
    flat_nn_parameters, unravel_pytree = flatten_util.ravel_pytree(neural_network_parameters)

    # FMU SETUP
    ####################################################################################
    fmu_filename = 'Van_der_Pol_damping_input.fmu'
    path = os.path.abspath(__file__)
    fmu_filename = '/'.join(path.split('/')[:-1]) + '/' + fmu_filename
    # Readout the model description and load the fmu into python
    fmu_evaluator = FMUEvaluator(fmu_filename, Tstart, Tend)

    # REFERENCE SOLUTION
    ####################################################################################
    z_ref = f_euler(z0=z0, t=t, fmu_evaluator=fmu_evaluator, model=damping, model_parameters=mu)
    fig = plt.figure()
    ax1, ax2 = fig.subplots(2, 1)
    ax1.plot(t, z_ref[:,0])
    ax1.plot(t, z_ref[:,1])
    ax1.set_title('Reference')
    # Reset and reinitialize the fmu for the next run after the reference run
    fmu_evaluator.reset_fmu(Tstart, Tend)

    # OPTIMISATION
    ####################################################################################
    epoch = 0

    # Vectorize the jacobian df_dtheta for all time points
    vectorized_df_dtheta_function = jax.jit(jax.vmap(df_dtheta_function, in_axes=(0, 0)))

    df_dz_function = lambda dfmu_dz, dinput_dz, dfmu_dinput: dfmu_dz + dinput_dz * dfmu_dinput
    vectorized_df_dz_function = jax.jit(jax.vmap(df_dz_function, in_axes=(0,0,0)))

    # Vectorize the jacobian dinput_dz for all time points (where input means neural network)
    dinput_dz_function = lambda p, z: jnp.array(jax.jacobian(neural_network.apply, argnums=1)(p, z))
    vectorized_dinput_dz_function = jax.jit(jax.vmap(dinput_dz_function, in_axes=(None, 0)))

    # Vectorize the jacobian dg_dtheta for all time points
    dg_dtheta_function = lambda z, z_ref, theta: jnp.array(jax.grad(g, argnums=2)(z, z_ref, theta))
    vectorized_dg_dtheta_function = jit(jax.vmap(dg_dtheta_function, in_axes=(0, 0, None)))

    dinput_dtheta_function = lambda p, z: jax.jacobian(neural_network.apply, argnums=0)(p, z)

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
            epoch]

    fmu_evaluator.training = True

    # Optimise the mu value via scipy
    # Optimisers CG, BFGS, Newton-CG, L-BFGS-B, TNC, SLSQP, dogleg, trust-ncg, trust-krylov, trust-exact and trust-constr
    res = minimize(optimisation_wrapper, flat_nn_parameters, method='BFGS', jac=True, args=args)
    print(res)
    # The values we optimized for are inside the result variable
    neural_network_parameters = res.x
    neural_network_parameters = unravel_pytree(neural_network_parameters)
    # Reset the FMU again for a final run for plotting purposes
    fmu_evaluator.reset_fmu(Tstart, Tend)


    # PLOT SOLUTION WITH REFERENCE
    ####################################################################################
    fmu_evaluator.training = False

    z, _, __ = f_euler(z0, t, fmu_evaluator, model=jitted_neural_network, model_parameters=neural_network_parameters)

    ax2.plot(t, z[:,0])
    ax2.plot(t, z[:,1])
    ax2.set_title('Solution')
    plt.show()
    fig.savefig('fmu_adjoint.png')

    path = os.path.abspath(__file__)
    plot_path = get_plot_path(path)
    plot_results(t, z, z_ref, plot_path)

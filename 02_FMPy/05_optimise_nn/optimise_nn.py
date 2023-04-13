# For interaction with the FMU
import fmpy
from fmpy import read_model_description, extract
from fmpy.fmi2 import _FMU2 as FMU2
from fmpy.util import plot_result, download_test_file
import ctypes
from typing import List

import pathlib
import re
import signal
import typing
from collections import OrderedDict
from enum import Enum
from types import SimpleNamespace

import fmpy
import numpy as np
import plac
import torch
import torch_optimizer as optim_contrib
import tqdm
# from modelexchange import FmuMEEvaluator, FmuMEModule
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.autograd import Function
from torch.autograd.function import once_differentiable
import os
from torch.optim import Optimizer

class FMUModule(nn.Module):
    fmu: FMU2

    def __init__(self, fmu, model_description):
        super().__init__()
        self.fmu = fmu
        self.model_description = model_description
        self.initialized = False
        self.tnow = 0.0

        # Will be set in initialize_fmu()
        self.Tstart = None
        self.Tend = None

        self._value_references = OrderedDict()
        self._input_value_references = OrderedDict()
        self._output_value_references = OrderedDict()
        self._state_value_references = OrderedDict()
        self._derivative_value_references = OrderedDict()
        for variable in model_description.modelVariables:
            self._value_references[variable.name] = variable.valueReference
            if variable.causality == "input":
                self._input_value_references[variable.name] = variable.valueReference
            if variable.causality == "output":
                self._output_value_references[variable.name] = variable.valueReference
            if variable.derivative:
                self._derivative_value_references[variable.name] = variable.valueReference
                self._state_value_references[
                    variable.derivative.name
                ] = variable.derivative.valueReference
        self._input_reference_numbers = list(self._input_value_references.values())
        self._output_reference_numbers = list(self._output_value_references.values())
        self._state_reference_numbers = list(self._state_value_references.values())
        self._derivative_reference_numbers = list(self._derivative_value_references.values())

        self.n_states = model_description.numberOfContinuousStates
        self.n_event_indicators = model_description.numberOfEventIndicators


        # Control the FMU simulation by setting parameters or determining
        # boundary values.
        self.physics_parameters = SimpleNamespace()

        # Example from the amesim code
        # self.physics_parameters = SimpleNamespace(switch_c=1)
        # self.physics_parameters.__dict__["mass_friction_endstops_1.xmin"] = 2.1
        # self.physics_parameters.__dict__["mass_friction_endstops_1.xmax"] = 2.1

    @property
    def state(self):
        return torch.tensor(self.fmu.getReal(self._state_reference_numbers))

    @property
    def output(self):
        return torch.tensor(self.fmu.getReal(self._output_reference_numbers))

    def forward(self, input, state):
        dx, y = FMUFunction.apply(
            input,
            state,
            self.fmu,
            self.tnow,
            self.pointers,
            self._input_reference_numbers,
            self._output_reference_numbers,
            self._derivative_reference_numbers,
            self._state_reference_numbers,
            self.training,
        )
        return dx, y

    def initialize_fmu(self, Tstart, Tend, value_references=None, values=None):
        self.fmu.setupExperiment(startTime=Tstart, stopTime=Tend)
        if value_references:
            self.fmu.setReal(value_references, values)
        self.fmu.enterInitializationMode()
        self.fmu.exitInitializationMode()

        self.initialEventMode = False
        self.enterEventMode = False
        self.timeEvent = False
        self.stateEvent = False
        self.previous_event_indicator = np.zeros(self.n_event_indicators)
        self.fmu.enterContinuousTimeMode()
        self.initialized = True

        # pointers to exchange state and derivative vectors with FMU
        self.pointers = SimpleNamespace(
            x=np.zeros(self.n_states),
            dx=np.zeros(self.n_states),
        )
        self.pointers._px = self.pointers.x.ctypes.data_as(
            ctypes.POINTER(ctypes.c_double)
        )
        self.pointers._pdx = self.pointers.dx.ctypes.data_as(
            ctypes.POINTER(ctypes.c_double)
        )

        self.Tstart = Tstart
        self.Tend = Tend

    def reset_fmu(self, Tstart, Tend, value_references=None, values=None):
        """Reset the FMU such that a new run can be started. New start and end times
        can be provided

        Parameters
        ----------
        Tstart : float
            Time from which to start the simulation
        Tend : _type_
            Time at which the simulation should be stopped

        Returns
        -------
        _type_
            _description_

        Raises
        ------
        RuntimeError
            _description_
        """

        self.fmu.reset()
        self.initialize_fmu(Tstart, Tend, value_references, values)



# Core of the combination of FMU and NN
# This is a special type of function which evaluates the FMU + augment model and returns
# the output and the gradient. For that it needs a forward and backward implementation
class FMUFunction(Function):
    @staticmethod
    def forward(ctx, u, x, *meta):
        fmu, tnow, pointers, input_reference, output_reference, derivative_references, state_references, training = meta

        dx, y = evaluate_FMU(u, x, fmu, tnow, pointers, input_reference, output_reference)

        if training:

            J_dxy_x = torch.zeros(len(derivative_references) + len(output_reference), len(state_references))
            J_dxy_u = torch.zeros(len(derivative_references) + len(output_reference), len(input_reference))

            for k in range(len(state_references)):
                J_dxy_x[:, k] = torch.tensor(
                    fmu.getDirectionalDerivative(derivative_references + output_reference, [state_references[k]], [1.0])
                )

            for k in range(len(input_reference)):
                J_dxy_u[:, k] = torch.tensor(
                    fmu.getDirectionalDerivative(derivative_references + output_reference, [input_reference[k]], [1.0])
                )

            ctx.save_for_backward(J_dxy_x, J_dxy_u)

        return dx, y

    @staticmethod
    @once_differentiable
    def backward(ctx, *grad_outputs):
        grad_dx, grad_y = grad_outputs

        grad_u = grad_x = None
        grad_meta = tuple([None] * 8)
        J_dxy_x, J_dxy_u = ctx.saved_tensors

        grad_u = torch.matmul(J_dxy_u.mT, torch.cat((grad_dx, grad_y), 0))
        grad_x = torch.matmul(J_dxy_x.mT, torch.cat((grad_dx, grad_y), 0))

        return grad_u, grad_x, *grad_meta


def evaluate_FMU(u, x, fmu, tnow, pointers, input_reference, output_reference):
    fmu.enterEventMode()
    newDiscreteStatesNeeded = True
    while newDiscreteStatesNeeded:
        (
            newDiscreteStatesNeeded,
            terminateSimulation,
            nominalsOfContinuousStatesChanged,
            valuesOfContinuousStatesChanged,
            nextEventTimeDefined,
            nextEventTime,
        ) = fmu.newDiscreteStates()
    fmu.enterContinuousTimeMode()

    # apply state
    pointers.x[:] = x.detach().numpy()
    fmu.setContinuousStates(pointers._px, pointers.x.size)

    # apply input
    fmu.setReal(input_reference, u.detach().tolist())

    # get state derivative
    fmu.getDerivatives(pointers._pdx, pointers.dx.size)

    fmu.setTime(tnow)
    step_event, _ = fmu.completedIntegratorStep()
    y = torch.tensor(fmu.getReal(output_reference))
    dx = torch.from_numpy(pointers.dx.astype(np.float32))

    return dx, y


class HybridModel(nn.Module):
    def __init__(
        self,
        fmu_module: FMUModule,
        augment_module: nn.Module,
        dt: float,
        solver: str = 'euler',
    ):
        super().__init__()
        self.fmu_module = fmu_module
        self.augment_module = augment_module
        self.dt = dt
        self.solver = solver
        self.Tstart = fmu_module.Tstart
        self.Tend = fmu_module.Tend

    def forward(self, control, augment_parameters: dict = {}):
        n_batches = control.shape[0]
        n_steps = control.shape[1]

        # Container to store trajectory
        X = torch.empty(n_batches, n_steps, len(self.fmu_module._state_reference_numbers))
        Y = torch.empty(n_batches, n_steps, len(self.fmu_module._output_reference_numbers))

                # custom_parameters contains all parameters for the FMU, which we want to
        # control from the outside. We collect them all now.
        # First add the physics_parameters.
        custom_parameters = OrderedDict(
            [
                (key, [value for _ in range(n_batches)])
                for key, value in self.physics_parameters.__dict__.items()
            ]
        )

        # Now we add the parameters from the augment_model; e.g. the mu parameter for VdP
        for key, value in augment_parameters.items():
            custom_parameters[key] = value

        assert all([len(x_) == n_batches for x_ in custom_parameters.values()])
        custom_parameter_value_references = [self.fmu_module._state_value_references[x_] for x_ in custom_parameters.keys()]

        for batch in range(n_batches):
            if not self.fmu_module.initialized:
                self.fmu_module.initialize_fmu(self.Tstart, self.Tend, custom_parameter_value_references, custom_parameters)
            x = self.fmu_module.state
            y = self.fmu_module.output

            for step in range(n_steps):
                if self.solver == 'euler':
                    u = self.augment_module(x)
                    X[batch, step, :] = x
                    Y[batch, step, :] = y
                    self.fmu_module.tnow += self.dt
                    dx, y = self.fmu_module(u, x)
                    x = x + self.dt*dx

            # terminate_fmu(self.fmu_module.fmu)

        return Y, X


class VdP(nn.Module):
    """Manages the augment model for the Reference solution. This means it
    contains the real term for the Van der Pol Oscillator which is missing
    in the FMU model

    Args:
        nn (_type_): _description_
    """
    def __init__(self) -> None:
        super().__init__()
        self.mu = nn.Parameter(torch.ones((1)))

    def forward(self, U):
        return self.mu * (1.0 - U[0]**2)*U[1]


def g(z, z_ref, ode_parameters):
    '''Calculates the inner part of the loss function.

    This function can either take individual floats for z
    and z_ref or whole numpy arrays'''
    return np.sum(0.5 * (z_ref - z)**2, axis = 0)

def J(z, z_ref, optimisation_parameters):
    '''Calculates the complete loss of a trajectory w.r.t. a reference trajectory'''
    return np.sum(g(z, z_ref, optimisation_parameters))


# For best interaction with the torch Neural Network we write the ajoint optimizer as
# a torch optimizer
class AdjointOptimizer(Optimizer):
    """Implements the adjoint optimization scheme for optimizing parameters in ODEs
    for Neural Networks"""

    def __init__(self, params):
        super(AdjointOptimizer, self).__init__(params)

    def _init_group(self, group, params_with_grad, grads):
        for p in group['params']:
            if p.grad is None:
                continue
            params_with_grad.append(p)
            if p.grad.is_sparse:
                raise RuntimeError("AdjointOptimizer does not support sparse gradients")
            grads.append(p.grad)


    def step(self, closure=None):
        """Performs a single optimization step

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []

            self._init_group(group, params_with_grad, grads)

            self.adjointoptimizer(
                params_with_grad,
                grads
            )

        return loss

    def adjointoptimizer(
        params: List[torch.Tensor],
        grads: List[torch.Tensor]
    ):

        pass

#
def instantiate_fmu(fmu_filename, Tstart, Tend):
    """Needs to be called before a FMUModule object can be created

    Parameters
    ----------
    fmu_filename : str
        The name of the FMU file
    Tstart : float
        Starting time for the FMU
    Tend : flaot
        Stopping time for the FMU

    Returns
    -------
    Tuple
        First element is the FMPy fmu object, second element is the model_description object
    """
    model_description = read_model_description(fmu_filename)
    # extract the FMU
    unzipdir = extract(fmu_filename)
    fmu = fmpy.fmi2.FMU2Model(guid=model_description.guid,
                    unzipDirectory=unzipdir,
                    modelIdentifier=model_description.modelExchange.modelIdentifier,
                    instanceName='instance1')
    # instantiate
    fmu.instantiate()
    return fmu, model_description



def terminate_fmu(fmu):
        fmu.terminate()
        fmu.freeInstance()

def optimisation_wrapper(optimisation_parameters, args):
    print(f'mu: {optimisation_parameters}')
    # t = args[0]
    # z0 = args[1]
    # z_ref = args[2]
    # fmu = args[3]
    # pointers = args[4]
    # number_of_states = args[5]
    # vr_derivatives = args[6]
    # vr_states = args[7]
    # vr_input = args[8]
    z_ref = args[2]
    model = args[3]
    U = args[4]
    z, _ = model(U)
    loss = J(z, z_ref)
    grad = loss.backward()



def simulate_custom_nn_input():
    fmu_filename = 'Van_der_Pol_damping_input.fmu'
    path = os.path.abspath(__file__)
    fmu_filename = '/'.join(path.split('/')[:-1]) + '/' + fmu_filename
    Tstart = 0.0
    Tend = 50.0
    nSteps = 2000
    dt = (Tend - Tstart)/(nSteps)
    Tspan = np.linspace(Tstart+dt, Tend, nSteps)
    solver = 'euler'

    fmu, model_description = instantiate_fmu(fmu_filename, Tstart, Tend)
    fmu_model = FMUModule(fmu, model_description)
    fmu_model.eval()

    augment_model = VdP()
    augment_model.mu.data[0] = torch.tensor(8.53)
    dummy_value = torch.tensor([2.0])

    # Create a dummy input since the VDP module does not need to be trained
    U = torch.zeros((len(dummy_value), Tspan.size, len(fmu_model._input_reference_numbers)))

    # Perform the reference run
    model = HybridModel(fmu_model, augment_model, dt, solver)

    with torch.no_grad():
        # Y, X = model(U, augment_parameters=OrderedDict(zip(["mu"], [x1_values])))
        Y, X = model(U)
    Yref = Y.detach()
    Xref = X.detach()

    augment_model = nn.Sequential(
        nn.Linear(2, 10),
        nn.Tanh(),
        # nn.Linear(1, 10),
        nn.Linear(10, 10),
        nn.Tanh(),
        nn.Linear(10, 1)
    )

    model = HybridModel(fmu_model, augment_model, dt, solver)
    loss_fcn = nn.MSELoss()
    optimizer = AdjointOptimizer(model.parameters())

    model.train()
    reset_fmu(fmu_model.fmu, fmu_model.model_description, Tstart, Tend)
    fmu_model.initialized = True

    Y, X = model(U)

    loss = loss_fcn(X, Xref)
    loss.retain_grad()
    test = loss.backward(retain_graph=True)
    # The gradients are in augment_model[0].weight.grad etc.
    print(test)
    optimizer.step()


if __name__ == '__main__':
    simulate_custom_nn_input()
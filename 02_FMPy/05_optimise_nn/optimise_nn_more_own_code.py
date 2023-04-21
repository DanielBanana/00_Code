import argparse
import os
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
# from torchdiffeq import odeint_adjoint, odeint_event
from torch.autograd import Function
from torch.autograd.function import once_differentiable

import fmpy
from fmpy import read_model_description, extract
from fmpy.fmi2 import _FMU2 as FMU2
from fmpy.util import plot_result, download_test_file
import ctypes
from typing import List

import numpy as np
from collections import OrderedDict
from types import SimpleNamespace
from typing import Sequence

class FMUModule(nn.Module):
    fmu: FMU2

    def __init__(self, fmu, model_description, Tstart=0.0, Tend=1.0):
        super().__init__()
        self.fmu = fmu
        self.model_description = model_description
        self.initialized = False
        self.tnow = 0.0

        # Will be set in initialize_fmu()
        self.Tstart = Tstart
        self.Tend = Tend

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

    # equivalent to function "f" in fmpy_masterfile_vdp_input
    def forward(self, u, x):
        # fmu = self.fmu
        # tnow = self.tnow
        # pointers = self.pointers
        # input_reference = self._input_reference_numbers
        # output_reference = self._output_reference_numbers
        # derivative_references = self._derivative_reference_numbers
        # state_references = self._state_reference_numbers
        # training = self.training
        # # fmu, tnow, pointers, input_reference, output_reference, derivative_references, state_references, training = meta

        # dx, y = evaluate_fmu(u, x, fmu, tnow, pointers, input_reference, output_reference)

        # if training:

        #     dfmu_dz_at_t = torch.zeros(len(derivative_references) + len(output_reference), len(state_references))
        #     dfmu_du_at_t = torch.zeros(len(derivative_references) + len(output_reference), len(input_reference))

        #     # Gradients of the function w.r.t. the ODE parameters (like position, velocity,...)
        #     for k in range(len(state_references)):
        #         dfmu_dz_at_t[:, k] = torch.tensor(
        #             fmu.getDirectionalDerivative(derivative_references + output_reference, [state_references[k]], [1.0])
        #         )
        #     # Gradients w.r.t. the control/input parameters like the output of a NN
        #     for k in range(len(input_reference)):
        #         dfmu_du_at_t[:, k] = torch.tensor(
        #             fmu.getDirectionalDerivative(derivative_references + output_reference, [input_reference[k]], [1.0])
        #         )

        #     # # These are for the gradients of the
        #     # ctx.save_for_backward(dfmu_dz_at_t, dfmu_du_at_t)
        #     # ctx.adjoint = 0
        # else:
        #     dfmu_dz_at_t = None
        #     dfmu_du_at_t = None

        # return dx, y, dfmu_dz_at_t, dfmu_du_at_t
        dx, y, dfmu_dz_at_t, dfmu_du_at_t = FMUFunction.apply(
            u,
            x,
            self.fmu,
            self.tnow,
            self.pointers,
            self._input_reference_numbers,
            self._output_reference_numbers,
            self._derivative_reference_numbers,
            self._state_reference_numbers,
            self.training,
        )
        return dx, y, dfmu_dz_at_t, dfmu_du_at_t

    def initialize_fmu(self, Tstart, Tend, value_references=None, values=None):
        self.fmu.setupExperiment(startTime=Tstart, stopTime=Tend)
        if value_references:
            self.fmu.setReal(value_references, values)
        self.fmu.enterInitializationMode()
        # set the input start values at time = Tstart
        pass
        self.fmu.exitInitializationMode()

        self.initialEventMode = False
        self.enterEventMode = False
        self.timeEvent = False
        self.stateEvent = False
        self.previous_event_indicator = np.zeros(self.n_event_indicators)
        self.fmu.enterContinuousTimeMode()

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

        self.initialized = True

    def reset_fmu(self):
        self.fmu.reset()
        self.initialized = False


# Core of the combination of FMU and NN
# This is a special type of function which evaluates the FMU + augment model and returns
# the output and the gradient. For that it needs a forward and backward implementation
# This is the equivalent to the function f in the file fmpy_masterfile_vdp_input.py
# located at 02_FMPy/04_optimise_mu


class FMUFunction(Function):
    @staticmethod
    def forward(ctx, u, x, *meta):
        fmu, tnow, pointers, input_reference, output_reference, derivative_references, state_references, training = meta

        dx, y = evaluate_fmu(u, x, fmu, tnow, pointers, input_reference, output_reference)

        if training:

            dfmu_dz_at_t = torch.zeros(len(derivative_references) + len(output_reference), len(state_references))
            dfmu_du_at_t = torch.zeros(len(derivative_references) + len(output_reference), len(input_reference))

            # Gradients of the function w.r.t. the ODE parameters (like position, velocity,...)
            for k in range(len(state_references)):
                dfmu_dz_at_t[:, k] = torch.tensor(
                    fmu.getDirectionalDerivative(derivative_references + output_reference, [state_references[k]], [1.0])
                )
            # Gradients w.r.t. the control/input parameters like the output of a NN
            for k in range(len(input_reference)):
                dfmu_du_at_t[:, k] = torch.tensor(
                    fmu.getDirectionalDerivative(derivative_references + output_reference, [input_reference[k]], [1.0])
                )

            # These are for the gradients of the
            ctx.save_for_backward(dfmu_dz_at_t, dfmu_du_at_t)
            ctx.adjoint = 0
        else:
            dfmu_dz_at_t = None
            dfmu_du_at_t = None

        return dx, y, dfmu_dz_at_t, dfmu_du_at_t

    @staticmethod
    @once_differentiable
    def backward(ctx, *grad_outputs):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output of the forward pass, and we need to compute the
        gradient of the loss with respect to the input of the forward pass.
        """
        grad_dx, grad_y, _, __ = grad_outputs
        ctx.adjoint +=1

        grad_u = grad_x = None
        grad_meta = tuple([None] * 8)
        dfmu_dz_at_t, dfmu_du_at_t = ctx.saved_tensors

        grad_u = torch.matmul(dfmu_du_at_t.mT, torch.cat((grad_dx, grad_y), 0))
        grad_x = torch.matmul(dfmu_dz_at_t.mT, torch.cat((grad_dx, grad_y), 0))

        return grad_u, grad_x, *grad_meta

def evaluate_fmu(u, x, fmu, tnow, pointers, input_reference, output_reference):
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
    dx = torch.from_numpy(pointers.dx.astype(np.float64))

    return dx, y

def instantiate_fmu(fmu_filename, Tstart, Tend):
    """Needs to be called before a FMUModule object can be created

    Parameters
    ----------
    fmu_filename : str
        The name of the FMU file
    Tstart : float
        Starting time for the FMU
    Tend : float
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


class HybridModel(nn.Module):
    def __init__(
        self,
        fmu_model: FMUModule,
        augment_model: nn.Module,
        dt: float,
        solver
    ):
        super().__init__()
        self.fmu_model = fmu_model
        self.augment_model = augment_model
        self.dt = dt
        self.Tstat = fmu_model.Tstart
        self.Tend = fmu_model.Tend
        self.solver = solver

    def forward(self, x):
        u = self.augment_model(x)
        dx, y, dfdx, dfdu = self.fmu_model(u, x)
        return dx, y, dfdx, dfdu

    # ode/hybrid_ode
    def simulate(self, x0, t):
        fmu_model = self.fmu_model
        augment_model = self.augment_model
        # fmu_model, augment_model, control, augment_parameters = meta
        # n_batches = control.shape[0]
        # n_steps = control.shape[1]
        n_steps = t.shape[0]
        if fmu_model.training:
            torch.set_grad_enabled(True)

        # Container to store trajectory
        X = torch.empty(n_steps, len(fmu_model._state_reference_numbers))
        Y = torch.empty(n_steps, len(fmu_model._output_reference_numbers))

        df_dz_trajectory = torch.empty(n_steps, 2, 2)
        df_du_trajectory = torch.empty(n_steps, 2, 1)

        # custom_parameters contains all parameters for the FMU, which we want to
        # control from the outside. We collect them all now.
        # First add the physics_parameters.
        n_batches = 1
        custom_parameters = OrderedDict(
            [
                (key, [value for _ in range(n_batches)])
                for key, value in fmu_model.physics_parameters.__dict__.items()
            ]
        )
        # # Now we add the parameters from the augment_model; e.g. the mu parameter for VdP
        # for key, value in augment_parameters.items():
        #     custom_parameters[key] = value
        assert all([len(x_) == n_batches for x_ in custom_parameters.values()])
        custom_parameter_value_references = [fmu_model._state_value_references[x_] for x_ in custom_parameters.keys()]

        if not fmu_model.initialized:
            fmu_model.initialize_fmu(fmu_model.Tstart, fmu_model.Tend, custom_parameter_value_references, custom_parameters)
        x0 = fmu_model.state
        y = fmu_model.output
        fmu_model.fmu.setReal(fmu_model._state_reference_numbers, x0.detach().numpy())
        if fmu_model.training:
            ans, dfdz, dfdz, t = HybridModelFunction.apply(x0, t, self.augment_model, self.fmu_model)
        else:
            ans = HybridModelFunction.apply(x0, t, self.augment_model, self.fmu_model)
        return ans

    # f_euler
    # def simulate(self, t, control, augment_parameters: dict = {}):
    #     """
    #     In the forward pass we receive a Tensor containing the input and return
    #     a Tensor containing the output. ctx is a context object that can be used
    #     to stash information for backward computation. You can cache arbitrary
    #     objects for use in the backward pass using the ctx.save_for_backward method.
    #     """
    #     # fmu_model, augment_model, control, augment_parameters = meta
    #     n_batches = control.shape[0]
    #     n_steps = control.shape[1]
    #     # ctx.n_batches = n_batches
    #     # ctx.n_steps = n_steps
    #     # ctx.n_states = len(fmu_model._state_reference_numbers)
    #     # ctx.t = t
    #     if self.fmu_model.training:
    #         torch.set_grad_enabled(True)

    #     # Container to store trajectory
    #     X = torch.empty(n_batches, n_steps, len(self.fmu_model._state_reference_numbers))
    #     Y = torch.empty(n_batches, n_steps, len(self.fmu_model._output_reference_numbers))
    #     df_dz_trajectory = torch.empty(n_batches, n_steps, 2, 2)
    #     df_du_trajectory = torch.empty(n_batches, n_steps, 2, 1)
    #     # custom_parameters contains all parameters for the FMU, which we want to
    #     # control from the outside. We collect them all now.
    #     # First add the physics_parameters.
    #     custom_parameters = OrderedDict(
    #         [
    #             (key, [value for _ in range(n_batches)])
    #             for key, value in self.fmu_model.physics_parameters.__dict__.items()
    #         ]
    #     )

    #     # Now we add the parameters from the augment_model; e.g. the mu parameter for VdP
    #     for key, value in augment_parameters.items():
    #         custom_parameters[key] = value

    #     assert all([len(x_) == n_batches for x_ in custom_parameters.values()])
    #     custom_parameter_value_references = [self.fmu_model._state_value_references[x_] for x_ in custom_parameters.keys()]

    #     for batch in range(n_batches):
    #         if not self.fmu_model.initialized:
    #             self.fmu_model.initialize_fmu(self.fmu_model.Tstart, self.fmu_model.Tend, custom_parameter_value_references, custom_parameters)
    #         x = self.fmu_model.state
    #         y = self.fmu_model.output

    #         for step in range(n_steps):
    #             X[batch, step, :] = x
    #             Y[batch, step, :] = y
    #             dx, y, df_dz_at_t, df_du_at_t = self(x)
    #             if self.fmu_model.training:
    #                 df_dz_trajectory[batch, step, :, :] = df_dz_at_t
    #                 df_du_trajectory[batch, step, :, :] = df_du_at_t
    #             x = x + self.dt*dx
    #     if self.fmu_model.training:
    #         pass
    #         # ctx.save_for_backward(df_dz_trajectory)
    #         # ctx.save_for_backward(df_du_trajectory)
    #     return X, Y, df_dz_trajectory, df_du_trajectory

class HybridModelFunction(torch.autograd.Function):

    @staticmethod
    def forward(x0, t, augment_model, fmu_model):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        X = torch.empty(len(t), len(fmu_model._state_reference_numbers))
        X[0] = x0
        x = x0
        df_dz_trajectory = torch.empty(len(t), 2, 2)
        df_du_trajectory = torch.empty(len(t), 2, 1)
        for step in range(len(t)-1):
            dt = t[step+1] - t[step]
            u = augment_model(x)
            dx, y, dfdx, dfdu = fmu_model(u, x)
            # dx, y, df_dz_at_t, df_du_at_t = hybrid_model(x)
            if fmu_model.training:
                df_dz_trajectory[step, :, :] = dfdx
                df_du_trajectory[step, :, :] = dfdu
            x = x + dt*dx
            X[step+1, :] = x
        if fmu_model.training:
            return X, df_dz_trajectory, df_du_trajectory, t
        else:
            return X

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        x0, t, augment_model, fmu_model = inputs
        if fmu_model.training:
            X, dfdz, dfdu, t = outputs
            ctx.save_for_backward(dfdz, dfdu, t)
        else:
            X = outputs



    @staticmethod
    def backward(ctx, grad_outputs):
        grad_dx = grad_outputs
        a = ctx.saved_tensors
        return None, grad_dx, None, None


# class NeuralNetwork(nn.Module):

#     def __init__(self, layers):
#         super().__init__()
#         self.layers = layers

#     def forward(x, parameters):
#         y = NeuralNetworkFunction.apply(x, parameters)

# class NeuralNetworkFunction(Function):
#     @staticmethod
#     def forward(ctx, x, parameters):

#         for layer in parameters:
#             a = activate(x, layer[0], layer[1])
#             x = a
#             ctx.save_for_backward(a)
#         return a

#     @staticmethod
#     def backward(ctx, *grad_outputs):
#         activations = ctx.saved_tensors
#         for activation in activations:
#             delta = activation * (1-activation) *



def activate(x, W, b):
    return 1./(1+torch.exp(-(W*x+b)))

# class HybridModelFunction(Function):
#     @staticmethod
#     def forward(ctx, t, *meta):
#         """
#         In the forward pass we receive a Tensor containing the input and return
#         a Tensor containing the output. ctx is a context object that can be used
#         to stash information for backward computation. You can cache arbitrary
#         objects for use in the backward pass using the ctx.save_for_backward method.
#         """
#         fmu_model, augment_model, control, augment_parameters = meta
#         n_batches = control.shape[0]
#         n_steps = control.shape[1]
#         ctx.n_batches = n_batches
#         ctx.n_steps = n_steps
#         ctx.n_states = len(fmu_model._state_reference_numbers)
#         ctx.t = t
#         if fmu_model.training:
#             torch.set_grad_enabled(True)

#         # Container to store trajectory
#         X = torch.empty(n_batches, n_steps, len(fmu_model._state_reference_numbers))
#         Y = torch.empty(n_batches, n_steps, len(fmu_model._output_reference_numbers))
#         df_dz_trajectory = torch.empty(n_batches, n_steps, 2, 2)
#         df_du_trajectory = torch.empty(n_batches, n_steps, 2, 1)
#         # custom_parameters contains all parameters for the FMU, which we want to
#         # control from the outside. We collect them all now.
#         # First add the physics_parameters.
#         custom_parameters = OrderedDict(
#             [
#                 (key, [value for _ in range(n_batches)])
#                 for key, value in fmu_model.physics_parameters.__dict__.items()
#             ]
#         )

#         # Now we add the parameters from the augment_model; e.g. the mu parameter for VdP
#         for key, value in augment_parameters.items():
#             custom_parameters[key] = value

#         assert all([len(x_) == n_batches for x_ in custom_parameters.values()])
#         custom_parameter_value_references = [fmu_model._state_value_references[x_] for x_ in custom_parameters.keys()]

#         for batch in range(n_batches):
#             if not fmu_model.initialized:
#                 fmu_model.initialize_fmu(fmu_model.Tstart, fmu_model.Tend, custom_parameter_value_references, custom_parameters)
#             x = fmu_model.state
#             y = fmu_model.output

#             for step in range(n_steps):
#                 u = augment_model(x)
#                 X[batch, step, :] = x
#                 Y[batch, step, :] = y
#                 dx, y, df_dz_at_t, df_du_at_t = fmu_model(u, x)
#                 if fmu_model.training:
#                     df_dz_trajectory[batch, step, :, :] = df_dz_at_t
#                     df_du_trajectory[batch, step, :, :] = df_du_at_t
#                 x = x + dt*dx
#         if fmu_model.training:
#             ctx.save_for_backward(df_dz_trajectory)
#             ctx.save_for_backward(df_du_trajectory)
#         return X, Y

#     @staticmethod
#     @once_differentiable
#     def backward(ctx, *grad_outputs):
#         """
#         In the backward pass we receive a Tensor containing the gradient of the loss
#         with respect to the output, and we need to compute the gradient of the loss
#         with respect to the input.
#         So we receive dJ/dX or dg/dX, as the main output from the forward function is X
#         We save df/dz and du/dz from the forward pass.
#         """
#         a0 = torch.zeros((2, 1))
#         df_dz_trajectory, df_du_trajectory = ctx.saved_tensors

#         # Now we need to flip the tensors we got because we integrate backwards in time
#         df_dz_trajectory = df_dz_trajectory.flip(1)
#         df_du_trajectory = df_du_trajectory.flip(1)
#         dg_dX_trajectory = grad_outputs[0].flip(1)
#         t = ctx.t.flip()

#         A = torch.empty(ctx.n_batches, ctx.n_steps, ctx.n_states)
#         A[0,0,:] = a0
#         for i in range(len(t)-1):
#             dt = t[i+1]-t[i]
#             A[0, i+1, :] += dt * (torch.matmul(-df_dz_trajectory[i].T, A[0, i, :]) - dg_dX_trajectory[i])
#         # Flip the backwards in time tensor so we get a forwards in time tensor
#         A = A.flip(1)


def g(z, z_ref, ode_parameters):
    '''Calculates the inner part of the loss function.

    This function can either take individual floats for z
    and z_ref or whole numpy arrays'''
    return torch.sum(0.5 * (z_ref - z)**2, axis = 0)

def J(z, z_ref, augment_parameters):
    return torch.sum(g(z, z_ref, augment_parameters))

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
        return self.mu # * (1.0 - U[0]**2)*U[1]


if __name__ == '__main__':

    torch.manual_seed(0)

    torch.set_default_dtype(torch.float64)

    fmu_filename = 'Van_der_Pol_input.fmu'
    path = os.path.abspath(__file__)
    fmu_filename = '/'.join(path.split('/')[:-1]) + '/' + fmu_filename
    Tstart = 0.0
    Tend = 50.0
    n_steps = 2000
    dt = (Tend - Tstart)/(n_steps)
    Tspan = np.linspace(Tstart, Tend, n_steps)
    solver = 'euler'
    x0 = torch.tensor([1.0, 0.0])

    fmu, model_description = instantiate_fmu(fmu_filename, Tstart, Tend)
    fmu_model = FMUModule(fmu, model_description, Tstart, Tend)
    fmu_model.eval()

    augment_model = VdP()
    augment_model.mu.data[0] = torch.tensor(8.53)
    dummy_value = torch.tensor([2.0])

    # Create a dummy input since the VDP module does not need to be trained
    U = torch.zeros((len(dummy_value), Tspan.size, len(fmu_model._input_reference_numbers)))

    # Perform the reference run
    model = HybridModel(fmu_model, augment_model, dt, solver)

    X = torch.empty(1, n_steps, len(fmu_model._state_reference_numbers), requires_grad=True)

    with torch.no_grad():
        # Y, X = model(U, augment_parameters=OrderedDict(zip(["mu"], [x1_values])))
        # (t, control, augment_parameters)
        X = model.simulate(x0, Tspan)
    # Yref = Y.detach()
    Xref = X.detach()

    model.fmu_model.reset_fmu()

    augment_model = nn.Sequential(
        nn.Linear(2, 10),
        nn.Tanh(),
        nn.Linear(10, 10),
        nn.Tanh(),
        nn.Linear(10, 1)
    )

    model = HybridModel(fmu_model, augment_model, dt, solver)
    fmu_model.train()
    augment_model.train()
    model.train()

    X = model.simulate(x0, Tspan)

    loss_fcn = nn.MSELoss()
    loss = loss_fcn(X, Xref)
    # for params in augment_model.parameters():
        # df_dtheta = torch.autograd.jacobian(model, params)
    # dg_dz = torch.autograd.grad(loss, X)
    loss.backward(retain_graph=True)
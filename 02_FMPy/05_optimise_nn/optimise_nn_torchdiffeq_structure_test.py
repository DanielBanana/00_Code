import argparse
import os
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

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

# Manages the solution of the FMU
class FMUModule(nn.Module):
    fmu: FMU2

    def __init__(self):
        super().__init__()

    def forward(self, input, state):
        dx, y = FMUFunction.apply(input, state)
        return dx, y

# Wraps the evaluation of the FMU from down below into a FMUFunction to allow the calculation
# of Gradients
class FMUFunction(Function):
    @staticmethod
    def forward(ctx, u, x, *meta):
        training = True
        dx, y = evaluate_FMU(u, x)

        if training:
            ctx.save_for_backward()

        return dx, y

    def backward(ctx, *grad_outputs):
        grad_dx, grad_y = grad_outputs
        grad_u = ctx.saved_tensors * grad_dx
        grad_x = ctx.saved_tensors * grad_dx
        return grad_u, grad_x, None

# Handles the pure FMU stuff, i.e. uses the FMPy interface to the fmu to just
# call the FMU with the inputs and gather the outputs
def evaluate_FMU(u, x):
    pass

augment_model = nn.Sequential(
    nn.Linear(2, 10),
    nn.Tanh(),
    nn.Linear(10, 10),
    nn.Tanh(),
    nn.Linear(10, 1)
)

class HybridModel(nn.Module):
    def __init__(
        self,
        fmu_module: FMUModule,
        augment_module: nn.Module
    ):
        self.fmu_module = fmu_module
        self.augment_module = augment_module

    # Perform one step of the combined model FMU + NN
    def forward(self, t, state):
        u = self.augment_module(state)
        dx, y = self.fmu_module(u, state)

    # calculate the whole trajectory of the combined model
    def simulate(self, times):
        odeint_adjoint(self, times)

# Wrapper Function for the ODE integration function for the combined model; checks parameters
# etc.
def odeint_adjoint(func, times):
    solution = OdeintAdjointMethod.apply(func, times)


class OdeintAdjointMethod(torch.autograd.Function):

    # Implement the whole ODE integration process here; save the solution trajectory and the
    # adjoint parameters in ctx
    @staticmethod
    def forward(ctx, func, times):
        sol = odeint(func, times)
        adjoint_parameters = []
        ctx.func = func
        ctx.save_for_backward(times, sol, *adjoint_parameters)

    @staticmethod
    def backward(ctx, *grad_y):
        times, solution, *adjoint_params = ctx.saved_tensors
        func = ctx.func
        def adjoint_dynamics(t, y_aug):
            y = y_aug[1]
            adj_y = y_aug[2]
            func_eval = func(t, y)
            return func_eval

        aug_state = odeint(adjoint_dynamics, aug_state)


# Performs ODE solution on provided function (use forward euler for example)
def odeint(func, times):
    pass
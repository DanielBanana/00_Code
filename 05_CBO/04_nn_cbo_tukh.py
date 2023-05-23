import os
import sys
import torch
import torch.nn.functional as F
import numpy as np


sys.path.insert(1, os.getcwd())
from fmu_helper import FMUEvaluator
from cbo_in_python.src.torch_.models import *
from cbo_in_python.src.datasets import load_mnist_dataloaders, load_parabola_dataloaders, f
from cbo_in_python.src.torch_.optimizer import Optimizer
from cbo_in_python.src.torch_.loss import Loss
from torch.utils.data import Dataset, DataLoader

MODELS = {
    'SimpleMLP': SimpleMLP,
    'TinyMLP': TinyMLP,
    'SmallMLP': SmallMLP,
    'LeNet1': LeNet1,
    'LeNet5': LeNet5,
}

# # The Neural Network structure class
# class ExplicitMLP(nn.Module):
#     features: Sequence[int]

#     def setup(self):
#         # we automatically know what to do with lists, dicts of submodules
#         self.layers = [nn.Dense(feat) for feat in self.features]
#         # for single submodules, we would just write:
#         # self.layer1 = nn.Dense(feat1)

#     def __call__(self, inputs):
#         x = inputs
#         for i, lyr in enumerate(self.layers):
#             x = lyr(x)
#             if i != len(self.layers) - 1:
#                 x = nn.relu(x)
#         return x

# For calculation of the reference solution we need the correct behaviour of the VdP
def damping(mu, inputs):
    return mu * (1 - inputs[0]**2) * inputs[1]

def _evaluate_reg(model, X_, y_, loss_fn):
    with torch.no_grad():
        outputs = model(X_)
        loss = loss_fn(outputs, y_)
    return loss

def f_euler(z0, t, fmu_evaluator: FMUEvaluator, model, model_parameters=None):
    '''Applies euler to the VdP ODE by calling the fmu; returns the trajectory'''
    z = np.zeros((t.shape[0], 2))
    z[0] = z0
    # Forward the initial state to the FMU
    fmu_evaluator.setup_initial_state(z0)
    times = []
    if fmu_evaluator.training:
        dfmu_dz_trajectory = []
        dfmu_dinput_trajectory = []
    for i in range(len(t)-1):
        # start = time.time()
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
        dfmu_dinput_trajectory = np.asarray(dfmu_dinput_trajectory)
        while len(dfmu_dinput_trajectory.shape) <= 2:
            dfmu_dinput_trajectory = np.expand_dims(dfmu_dinput_trajectory, -1)
        return z, np.asarray(dfmu_dz_trajectory), np.asarray(dfmu_dinput_trajectory)
    else:
        return z


if __name__ == '__main__':
    # ODE SETUP
    ####################################################################################
    Tstart = 0.0
    Tend = 10.0
    nSteps = 1001
    t = np.linspace(Tstart, Tend, nSteps)
    z0 = np.array([1.0, 0.0])
    mu = 5.0

    # FMU SETUP
    ####################################################################################
    fmu_filename = 'Van_der_Pol_damping_input.fmu'
    path = os.path.abspath(__file__)
    fmu_filename = '/'.join(path.split('/')[:-1]) + '/' + fmu_filename
    fmu_evaluator = FMUEvaluator(fmu_filename, Tstart, Tend)

    z_ref = f_euler(z0=z0, t=t, fmu_evaluator=fmu_evaluator, model=damping, model_parameters=mu)
    fmu_evaluator.reset_fmu(Tstart, Tend)

    # CONVERT THE REFERENCE DATA TO A DATASET
    ####################################################################################
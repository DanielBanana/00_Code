from fmu_helper import FMUEvaluator

# General
import numpy as np
from matplotlib import pyplot as plt
import os
import sys
import time

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

        # if fmu_evaluator.training:
        #     derivatives, enterEventMode, terminateSimulation, dfmu_dz_at_t, dfmu_dinput_at_t = fmu_evaluator.evaluate_fmu(t[i], dt, model, model_parameters)
            # dfmu_dz_trajectory.append(dfmu_dz_at_t)
            # dfmu_dinput_trajectory.append(dfmu_dinput_at_t)
        # else:
        #     derivatives, enterEventMode, terminateSimulation = fmu_evaluator.evaluate_fmu(t[i], dt, model, model_parameters)

        derivatives, enterEventMode, terminateSimulation = fmu_evaluator.evaluate_fmu(t[i], dt, model, model_parameters)

        z[i+1] = z[i] + dt * derivatives

        if terminateSimulation:
            break

    # We get on jacobian less then we get datapoints, since we get the jacobian
    # with every derivative we calculate, and we have one datapoint already given
    # at the start
    # if fmu_evaluator.training:
    #     derivatives, enterEventMode, terminateSimulation, dfmu_dz_at_t, dfmu_dinput_at_t = fmu_evaluator.evaluate_fmu(t[i], dt, model, model_parameters)
    #     # dfmu_dz_trajectory.append(dfmu_dz_at_t)
    #     # dfmu_dinput_trajectory.append(dfmu_dinput_at_t)
    #     # dfmu_dinput_trajectory = jnp.asarray(dfmu_dinput_trajectory)
    #     # while len(dfmu_dinput_trajectory.shape) <= 2:
    #     #     dfmu_dinput_trajectory = jnp.expand_dims(dfmu_dinput_trajectory, -1)
    #     return z, np.asarray(dfmu_dz_trajectory), jnp.asarray(dfmu_dinput_trajectory)
    # else:
        # return z
    return z


if __name__ == '__main__':
    # ODE SETUP
    ####################################################################################
    Tstart = 0.0
    Tend = 1.0
    nSteps = 2
    t = np.linspace(Tstart, Tend, nSteps)
    z0 = np.array([1.0, 0.0])

    name = "NN"
    path = os.path.abspath(__file__)
    directory = '/'.join(path.split('/')[:-1])
    path  = os.path.join(directory, f"{name}", "build", f"{name}.fmu")

    fmu_evaluator = FMUEvaluator(path, Tstart, Tend)

    output = fmu_evaluator.evaluate_nn_fmu(t=Tend, inputs = [1.0, 0.0])

    fmu_evaluator.reset_fmu(Tstart, Tend)
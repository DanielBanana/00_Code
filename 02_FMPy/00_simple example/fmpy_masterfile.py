""" This example demonstrates how to use the FMU.get*() and FMU.set*() functions
 to set custom input and control the simulation """
import fmpy
from fmpy import read_model_description, extract
from fmpy.fmi2 import _FMU2 as FMU2
from fmpy.util import plot_result, download_test_file
import numpy as np
import shutil
import ctypes
from types import SimpleNamespace
from matplotlib import pyplot as plt
import os

def settable_in_instantiated(variable):
    return variable.causality == 'input' \
           or variable.variability != 'constant' and variable.initial in {'approx', 'exact'}

def simple_example():
    # define the model name and simulation parameters
    fmu_filename = 'simple_example.fmu'
    path = os.path.abspath(__file__)
    fmu_filename = '/'.join(path.split('/')[:-1]) + '/' + fmu_filename
    Tstart = 0.0
    Tend = 2.0
    nSteps = 100
    dt = (Tend - Tstart)/(nSteps)
    Tspan = np.linspace(Tstart+dt, Tend, 100)

    # Readout the model description and load the fmu into python
    model_description = read_model_description(fmu_filename)
    unzipdir = extract(fmu_filename)
    fmu = fmpy.fmi2.FMU2Model(guid=model_description.guid,
                    unzipDirectory=unzipdir,
                    modelIdentifier=model_description.modelExchange.modelIdentifier,
                    instanceName='instance1')
    eventInfo = fmpy.fmi2.fmi2EventInfo()

    # instantiate, always needs to happen at the start
    fmu.instantiate()

    # set the start time
    time = Tstart

    # set variable start values (of "ScalarVariable / <type> / start")
    pass

    # initialize, needs to happen to set start values
    # determine continous and discrete states
    fmu.setupExperiment(startTime=Tstart, stopTime=Tend)
    fmu.enterInitializationMode()

    # set the input start values at time = Tstart
    pass

    fmu.exitInitializationMode()

    # Prepare working with the fmu
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
        z=np.zeros(nz)
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
    vr_inputs   = vrs['x']
    vr_derivatives = vrs['der(x)']

    fmu.enterContinuousTimeMode()
    status = fmu.getContinuousStates(pointers._px, pointers.x.size)
    x_history = [pointers.x[0]]

    for time in Tspan:
        # handle events
        if initialEventMode or enterEventMode or timeEvent or stateEvent:
            if not initialEventMode:
                fmu.enterEventMode()
            # event iteration
            pass

            # enter Continuous-Time Mode
            fmu.enterContinuousTimeMode()

            # retrieve solution at simulation (re)start
            pass

            # if initialEventMode or valuesOfContinuousStatesChanged:
            # the model signals a value change of states, retrieve them
            # In this simple example we don't need to check that; it changes every iteration
            status = fmu.getContinuousStates(pointers._px, pointers.x.size)

        if time >= Tend:
            break
        # compute derivatives
        status = fmu.getDerivatives(pointers._pdx, pointers.dx.size)

        # advance time
        status = fmu.setTime(time)

        # set continuous inputs at t = time
        pass

        # set states at t = time and perform one setp
        pointers.x[0] = pointers.x[0] + dt * pointers.dx[0]
        x_history.append(pointers.x[0])
        status = fmu.setContinuousStates(pointers._px, pointers.x.size)

        # get event indicators at t = time
        status = fmu.getEventIndicators(pointers._pz, pointers.z.size)

        # Process to get gradients for optimisation; not needed now but for optimisation
        df_dx = fmu.getDirectionalDerivative([vr_derivatives], [vr_inputs], [1.0])


        # inform the model about an accepted step
        enterEventMode, terminateSimulation = fmu.completedIntegratorStep()

        # get continuous output
        # fmu.getReal([vr_outputs])

        if terminateSimulation:
            break

    print('ODE: dx/dt = 10*x')
    print(f'Directional Derivative (Jacobian) calculated: {df_dx}')
    fig = plt.figure()
    ax = fig.subplots()
    ax.plot(Tspan, x_history)
    plt.show()
    fig.savefig('simple_example.png')


if __name__ == '__main__':
    simple_example()
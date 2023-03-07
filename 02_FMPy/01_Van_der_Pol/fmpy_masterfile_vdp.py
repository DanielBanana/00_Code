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

def settable_in_instantiated(variable):
    return variable.causality == 'input' \
           or variable.variability != 'constant' and variable.initial in {'approx', 'exact'}

def van_der_pol():

    # define the model name and simulation parameters
    fmu_filename = 'Van_der_Pol.fmu'
    Tstart = 0.0
    Tend = 500.0
    nSteps = 100000
    _lambda = 8.53
    dt = (Tend - Tstart)/(nSteps)
    Tspan = np.linspace(Tstart+dt, Tend, nSteps)
    model_description = read_model_description(fmu_filename)
    # extract the FMU
    unzipdir = extract(fmu_filename)
    fmu = fmpy.fmi2.FMU2Model(guid=model_description.guid,
                    unzipDirectory=unzipdir,
                    modelIdentifier=model_description.modelExchange.modelIdentifier,
                    instanceName='instance1')
    eventInfo = fmpy.fmi2.fmi2EventInfo()

    # instantiate
    fmu.instantiate()

    # set the start time
    time = Tstart

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
    vr_inputs   = [vrs['u'], vrs['v']]
    vr_derivatives = [vrs['der(u)'], vrs['der(v)']]
    vr_parameter = vrs['mu']

    fmu.setReal([vr_parameter], [_lambda])

    # retrieve solution at t=Tstart, for example, for outputs
    # y = fmu.getReal([vr_outputs])

    fmu.enterContinuousTimeMode()

    x_history = [pointers.x.copy()]

    for i in range(1, len(Tspan)):
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
        time = Tspan[i]
        status = fmu.setTime(time)

        # set continuous inputs at t = time
        pass

        # set states at t = time and perform one setp
        pointers.x += dt * pointers.dx
        x_history.append(pointers.x.copy())
        status = fmu.setContinuousStates(pointers._px, pointers.x.size)

        # get event indicators at t = time
        status = fmu.getEventIndicators(pointers._pz, pointers.z.size)

        df_dx = np.zeros((2,2))
        for j in range(nx):
            df_dx[:, j] = np.array(fmu.getDirectionalDerivative(vr_derivatives, [vr_inputs[j]], [1.0]))


        # detect events, if any
        # timeEvent = time >= tNext
        # stateEvent = sign(z) <> sign(previous_z) or previous_z != 0 && z == 0
        # previous_z = z

        # inform the model about an accepted step
        enterEventMode, terminateSimulation = fmu.completedIntegratorStep()

        # get continuous output
        # fmu.getReal([vr_outputs])

        if terminateSimulation:
            break
    
    # print('ODE: dx/dt = 10*x')
    x_history = np.asarray(x_history).T

    print(f'Directional Derivative (Jacobian) calculated at t={Tspan[i]}: {df_dx}')
    fig = plt.figure()
    ax, ax2 = fig.subplots(2,1)
    ax.plot(Tspan, x_history[0])
    ax2.plot(Tspan, x_history[1])
    plt.show()
    fig.savefig('Van_der_Pol.png')


    # J = fmu.getDirectionalDerivative(vUnknown_ref=[vr_derivatives], vKnown_ref=[vr_inputs], dvKnown=[1.0])


if __name__ == '__main__':

    van_der_pol()
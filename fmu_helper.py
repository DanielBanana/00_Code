# For interaction with the FMU
import fmpy
from fmpy import read_model_description, extract
from fmpy.fmi2 import _FMU2 as FMU2
import ctypes
from types import SimpleNamespace
from typing import List
import numpy as np

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
        # pointers = SimpleNamespace(
        #     x=np.zeros(self.n_states),
        #     dx=np.zeros(self.n_states),
        #     z=np.zeros(self.n_events),
        # )
        # pointers._px = pointers.x.ctypes.data_as(
        #     ctypes.POINTER(ctypes.c_double)
        # )
        # pointers._pdx = pointers.dx.ctypes.data_as(
        #     ctypes.POINTER(ctypes.c_double)
        # )
        # pointers._pz = pointers.z.ctypes.data_as(
        #     ctypes.POINTER(ctypes.c_double)
        # )
        # status = self.fmu.getContinuousStates(pointers._px, pointers.x.size)
        # self.pointers = pointers

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

        # # retrieve initial state x and
        # # nominal values of x (if absolute tolerance is needed)
        # # pointers to exchange state and derivative vectors with FMU
        # pointers = SimpleNamespace(
        #     x=np.zeros(self.n_states),
        #     dx=np.zeros(self.n_states),
        #     z=np.zeros(self.n_events),
        # )
        # pointers._px = pointers.x.ctypes.data_as(
        #     ctypes.POINTER(ctypes.c_double)
        # )
        # pointers._pdx = pointers.dx.ctypes.data_as(
        #     ctypes.POINTER(ctypes.c_double)
        # )
        # pointers._pz = pointers.z.ctypes.data_as(
        #     ctypes.POINTER(ctypes.c_double)
        # )
        # self.pointers = pointers

        self.fmu.enterContinuousTimeMode()

    def evaluate_fmu(self, t, dt, augment_model_function, augment_model_args, pointers):
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
            control = augment_model_function(augment_model_args, pointers.x)
            self.fmu.setReal(self.vr_input, [control])
            status = self.fmu.getDerivatives(pointers._pdx, pointers.dx.size)
            dfmu_dz_at_t = self.dfmu_dz_function()
            dfmu_dinput_at_t = self.dfmu_dinput_function()
        else:
            control = augment_model_function(augment_model_args, pointers.x)
            self.fmu.setReal(self.vr_input, [control])
            status = self.fmu.getDerivatives(pointers._pdx, pointers.dx.size)

        pointers.x += dt * pointers.dx

        status = self.fmu.setContinuousStates(pointers._px, pointers.x.size)

        # get event indicators at t = time
        status = self.fmu.getEventIndicators(pointers._pz, pointers.z.size)

        # inform the model about an accepted step
        enterEventMode, terminateSimulation = self.fmu.completedIntegratorStep()

        # get continuous output
        # fmu.getReal([vr_outputs])

        # If we are in Training mode return the derivatives for the next step,
        # FMU information and Optimisation jacobians; otherwise leave out jacobians
        if self.training:
            return pointers, enterEventMode, terminateSimulation, dfmu_dz_at_t, dfmu_dinput_at_t
        else:
            return pointers, enterEventMode, terminateSimulation

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

    def setup_initial_state(self, z0, pointers):
        """Before starting the iteration of the ODE solver set the inital state in the
        FMU and load the pointers with the correct values

        Parameters
        ----------
        z0 : _type_
            _description_
        """
        self.fmu.setReal(self.vr_states, z0)
        self.fmu.getContinuousStates(pointers._px, pointers.x.size)


    def get_pointers(self):
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
        return pointers
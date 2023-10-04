# For interaction with the FMU
import fmpy
from fmpy import read_model_description, extract
from fmpy.fmi2 import _FMU2 as FMU2
import ctypes
from types import SimpleNamespace
from typing import List
import numpy as np
from copy import deepcopy

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
        # self.model_description_file = "/".join(fmu_filename.split("/")[:-1]+["modelDescription.xml"])

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
        self.vr_states = []
        self.vr_derivatives = []
        self.vr_inputs = []
        self.vr_outputs = []
        for variable in self.model_description.modelVariables:
            vrs[variable.name] = variable.valueReference
            if variable.causality == 'local':
                if variable.initial == 'exact':
                    self.vr_states.append(variable.valueReference)
                elif variable.initial == 'calculated':
                    self.vr_derivatives.append(variable.valueReference)
            elif variable.causality == 'input':
                self.vr_inputs.append(variable.valueReference)
            elif variable.causality == 'output':
                self.vr_outputs.append(variable.valueReference)

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

    def evaluate_nn_fmu(self, t, inputs):
        self.fmu.setTime(t)
        self.fmu.setReal(vr=self.vr_inputs, value=inputs)
        # self.fmu.getDerivatives(self.pointers._pdx, self.pointers.dx.size)
        # self.fmu.getEventIndicators(self.pointers._pz, self.pointers.z.size)
        enterEventMode, terminateSimulation = self.fmu.completedIntegratorStep()


        outputs = self.fmu.getReal(self.vr_outputs)

        return outputs


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
        self.fmu.getContinuousStates(self.pointers._px, self.pointers.x.size)

        if self.training:
            if augment_model_function is not None:
                control = augment_model_function(params=augment_model_args, inputs=self.pointers.x)
                self.fmu.setReal(self.vr_inputs, control if type(control) is list else [control])
            status = self.fmu.getDerivatives(self.pointers._pdx, self.pointers.dx.size)
            dfmu_dz_at_t = self.dfmu_dz_function()
            dfmu_dinput_at_t = self.dfmu_dinput_function()
        else:
            if augment_model_function is not None:
                # a = deepcopy(self.pointers.x)
                control = augment_model_function(params=augment_model_args, inputs=self.pointers.x)
                # self.pointers.x = a
                self.fmu.setReal(self.vr_inputs, control if type(control) is list else [control])
            status = self.fmu.getDerivatives(self.pointers._pdx, self.pointers.dx.size)

        self.pointers.x += dt * self.pointers.dx

        status = self.fmu.setContinuousStates(self.pointers._px, self.pointers.x.size)

        # get event indicators at t = time
        status = self.fmu.getEventIndicators(self.pointers._pz, self.pointers.z.size)

        # inform the model about an accepted step
        enterEventMode, terminateSimulation = self.fmu.completedIntegratorStep()

        # get continuous output
        # outputs = self.fmu.getReal(self.vr_outputs)

        # If we are in Training mode return the derivatives for the next step,
        # FMU information and Optimisation jacobians; otherwise leave out jacobians
        if self.training:
            return enterEventMode, terminateSimulation, dfmu_dz_at_t, dfmu_dinput_at_t
        else:
            return enterEventMode, terminateSimulation

    def get_derivatives(self, t, state):
            """Function to just evaluate the FMU with no contribution of the augment model.
            Uses the internally stored FMU to calculate the derivatives. Does not progress the FMU

            Parameters
            ----------
            t : float
                The current time
            dt : _type_
                The current time step

            Returns
            -------
            _type_
                Derivatives for the new state, Flag whether event mode has been entered,
                Flag whether the simulation needs to be terminated; In Training mode:
                dfmu_dz, dfmu_dinput
            """
            status = self.fmu.setTime(t)

            x = self.pointers.x
            dx = list(self.pointers.dx)

            self.pointers.x = state

            if self.training:
                control = 0.0
                self.fmu.setReal(self.vr_inputs, [control])
                status = self.fmu.getDerivatives(self.pointers._pdx, self.pointers.dx.size)
                dfmu_dz_at_t = self.dfmu_dz_function()
                dfmu_dinput_at_t = self.dfmu_dinput_function()
            else:
                control = 0.0
                self.fmu.setReal(self.vr_inputs, [control])
                status = self.fmu.getDerivatives(self.pointers._pdx, self.pointers.dx.size)

            tmp_dx = self.pointers.dx
            self.pointers.dx = np.asarray(dx)
            self.pointers.x = x

            # get continuous output
            # fmu.getReal([vr_outputs])

            # If we are in Training mode return the derivatives for the next step,
            # FMU information and Optimisation jacobians; otherwise leave out jacobians
            if self.training:
                return tmp_dx, dfmu_dz_at_t, dfmu_dinput_at_t
            else:
                return tmp_dx

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
        dfmu_dz = np.zeros((self.n_states,self.n_states))
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
        return self.fmu.getDirectionalDerivative(self.vr_derivatives, self.vr_inputs, [1.0])

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



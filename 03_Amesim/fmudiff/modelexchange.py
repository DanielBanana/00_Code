import ctypes
import time
from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import torch
from fmpy.fmi2 import _FMU2 as FMU2
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable

torch.set_default_dtype(torch.float64)


class FmuMEEvaluator(object):
    @staticmethod
    def evaluate(u, x, fmu, tnow, pointers, ru, ry):
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
        fmu.setReal(ru, u.detach().tolist())

        # get state derivative
        fmu.getDerivatives(pointers._pdx, pointers.dx.size)

        fmu.setTime(tnow)
        step_event, _ = fmu.completedIntegratorStep()
        y = torch.tensor(fmu.getReal(ry))
        dx = torch.from_numpy(pointers.dx.astype(np.float64))

        return dx, y


class FmuMEFunction(Function):
    @staticmethod
    def forward(ctx, u, x, *meta):
        fmu, tnow, pointers, ru, ry, rdx, rx, training = meta

        dx, y = FmuMEEvaluator.evaluate(u, x, fmu, tnow, pointers, ru, ry)

        if training:

            J_dxy_x = torch.zeros(len(rdx) + len(ry), len(rx))
            J_dxy_u = torch.zeros(len(rdx) + len(ry), len(ru))

            for k in range(len(rx)):
                J_dxy_x[:, k] = torch.tensor(
                    fmu.getDirectionalDerivative(rdx + ry, [rx[k]], [1.0])
                )

            for k in range(len(ru)):
                J_dxy_u[:, k] = torch.tensor(
                    fmu.getDirectionalDerivative(rdx + ry, [ru[k]], [1.0])
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


class FmuMEModule(nn.Module):
    fmu: FMU2

    def __init__(self, fmu, model_description, verbose=False, logging=False):
        super().__init__()
        self.fmu = fmu
        self.initialized = False
        self.tnow = 0.0
        self.fmu.verbose = verbose
        self.fmu.logging = logging
        self._vrs = OrderedDict()
        self._vrsu = OrderedDict()
        self._vrsy = OrderedDict()
        self._vrsx = OrderedDict()
        self._vrsdx = OrderedDict()
        for variable in model_description.modelVariables:
            self._vrs[variable.name] = variable.valueReference
            if variable.causality == "input":
                self._vrsu[variable.name] = variable.valueReference
            if variable.causality == "output":
                self._vrsy[variable.name] = variable.valueReference
            if variable.derivative:
                self._vrsdx[variable.name] = variable.valueReference
                self._vrsx[
                    variable.derivative.name
                ] = variable.derivative.valueReference
        self._ru = list(self._vrsu.values())
        self._ry = list(self._vrsy.values())
        self._rx = list(self._vrsx.values())
        self._rdx = list(self._vrsdx.values())
        nx = model_description.numberOfContinuousStates
        # pointers to exchange state and derivative vectors with FMU
        self.pointers = SimpleNamespace(
            x=np.zeros(nx),
            dx=np.zeros(nx),
        )
        self.pointers._px = self.pointers.x.ctypes.data_as(
            ctypes.POINTER(ctypes.c_double)
        )
        self.pointers._pdx = self.pointers.dx.ctypes.data_as(
            ctypes.POINTER(ctypes.c_double)
        )

    @property
    def state(self):
        return torch.tensor(self.fmu.getReal(self._rx))

    @property
    def output(self):
        return torch.tensor(self.fmu.getReal(self._ry))

    def forward(self, u, x):
        dx, y = FmuMEFunction.apply(
            u,
            x,
            self.fmu,
            self.tnow,
            self.pointers,
            self._ru,
            self._ry,
            self._rdx,
            self._rx,
            self.training,
        )
        return dx, y

    def fmu_initialize(self, rv=None, v=None):
        if self.initialized:
            return
        FMU2.__init__(
            self.fmu,
            guid=self.fmu.guid,
            modelIdentifier=self.fmu.modelIdentifier,
            unzipDirectory=self.fmu.unzipDirectory,
            instanceName=self.fmu.instanceName,
            libraryPath=self.fmu.dll._name,
            fmiCallLogger=self.fmu.fmiCallLogger,
        )
        self.fmu.instantiate(
            visible=self.fmu.verbose, callbacks=None, loggingOn=self.fmu.logging
        )
        self.fmu.setupExperiment(startTime=0.0, stopTime=None)
        if rv:
            self.fmu.setReal(rv, v)
        self.fmu.enterInitializationMode()
        self.fmu.exitInitializationMode()
        self.initialized = True
        self.tnow = 0.0

    def fmu_terminate(self):
        if not self.initialized:
            return
        self.fmu.terminate()
        self.fmu.freeInstance()
        self.initialized = False

    def fmu_reinitialize(self, rv=None, v=None):
        self.fmu_terminate()
        time.sleep(0.01)
        self.fmu_initialize(rv, v)
        self.initialized = True

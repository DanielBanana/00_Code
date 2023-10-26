import torch
import torch.nn as nn

import numpy as np

import copy
from copy import deepcopy


class Particle(nn.Module):
    def __init__(self, model, fmu=False, residual=False, restore=False):
        """
        Represents a particles in the consensus-based optimization. Stores a copy of the optimized model.
        :param model: the underlying model.
        """
        super(Particle, self).__init__()
        self.fmu = fmu
        self.residual = residual

        if fmu:
            self.pointers = model.fmu_model.get_pointers()
            fmu_model = model.fmu_model
            model.fmu_model = None
            self.model = copy.copy(model)
            if restore:
                for p in self.model.parameters():
                    with torch.no_grad():
                        p.copy_(p + 0.01*torch.randn_like(p))
            else:
                for p in self.model.parameters():
                    with torch.no_grad():
                        p.copy_(torch.randn_like(p))
            self.model.fmu_model = fmu_model
            model.fmu_model = fmu_model
        else:
            self.model = deepcopy(model)
            model_copy = deepcopy(model.augment_model.model)
            if restore:
                for p in model_copy.parameters():
                    with torch.no_grad():
                        p.copy_(p + 0.01*torch.randn_like(p))
            else:
                for p in model_copy.parameters():
                    with torch.no_grad():
                        p.copy_(torch.randn_like(p))

            self.model.augment_model.model = model_copy

    def forward(self, X):
        if self.fmu:
            output = self.model(self.pointers)
            if type(output) == np.ndarray:
                output = torch.tensor(output)
            return output
        else:
            if self.residual:
                return self.model(X)
            else:
                output = self.model()
                if type(output) == np.ndarray:
                    output = torch.tensor(output)
                return output
    def get_params(self):
        """
        :return: the underlying models' parameters stacked into a 1d-tensor.
        """
        return torch.cat([p.view(-1) for p in self.model.parameters()]).view(-1)

    def set_params(self, new_params):
        """
        Updates the underlying models' parameters.
        :param new_params: new params stacked into a 1d-tensor.
        """
        next_slice = 0
        for p in self.model.parameters():
            slice_length = len(p.view(-1))
            with torch.no_grad():
                p.copy_(new_params[next_slice: next_slice + slice_length].view(p.shape))
            next_slice += slice_length

    def get_gradient(self):
        """
        Returns the gradients stacked into a 1d-tensor.
        """
        gradients = [p.grad for p in self.model.parameters()]
        if None in gradients:
            return None
        return torch.cat([g.view(-1) for g in gradients]).view(-1)

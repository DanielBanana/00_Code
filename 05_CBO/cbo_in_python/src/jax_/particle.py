import torch
import torch.nn as nn

import numpy as np

from copy import deepcopy

import jax
import jax.numpy as jnp
import jax.random as jrandom


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
            self.model = deepcopy(model)
            if self.model.restore:
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
            key, key2 = jrandom.split(jrandom.PRNGKey(np.random.randint(0,99999)))
            ps = []
            if self.model.restore:
                randinit = lambda x: x + 0.01*jrandom.normal(key, x.shape)
                self.model.nn_parameters = jax.tree_util.tree_map(randinit, model.nn_parameters)
                # for p in self.model.parameters():
                #     # with torch.no_grad():
                #     key1, key2 = jrandom.split(key2)
                #     p = jnp.copy(p + 0.01*jrandom.normal(key1, p.shape))
                #     ps.append(p)
                # self.model.set_parameters(ps)
            else:
                randinit = lambda x: jrandom.normal(key, x.shape)
                self.model.nn_parameters = jax.tree_util.tree_map(randinit, model.nn_parameters)
                # for p in self.model.parameters():
                #     # with torch.no_grad():
                #     key1, key2 = jrandom.split(key2)
                #     p = jnp.copy(jrandom.normal(key1, p.shape))
                #     ps.append(p)
                # self.model.set_parameters(ps)

    def forward(self, X):
        if self.fmu:
            return self.model(self.pointers)
            # if type(output) == np.ndarray:
            #     output = torch.tensor(output)
            # return output
        else:
            if self.residual:
                return self.model(X)
            else:
                return self.model(X)
                # if type(output) == np.ndarray:
                #     output = torch.tensor(output)
                # return output

    def get_params(self):
        """
        :return: the underlying models' parameters stacked into a 1d-tensor.
        """
        # return torch.cat([p.view(-1) for p in self.model.parameters()]).view(-1)
        # return jnp.concatenate([p.reshape(-1) for p in self.model.parameters()]).reshape(-1)
        # params, unravel = jax.flatten_util.ravel_pytree(self.model.nn_parameters)
        # import time
        # start = time.time()
        # parmeters = self.model.parameters_flat()
        # print(f'flat: {time.time()-start}')
        # start = time.time()
        # parameters = self.model.parameters()
        # print(f'nested: {time.time()-start}')
        # return self.model.parameters_flat() # 0.07s
        return self.model.parameters() # 0.0001s

    def set_params(self, new_params):
        """
        Updates the underlying models' parameters.
        :param new_params: new params stacked into a 1d-tensor.
        """
        if new_params is list:
            self.model.set_parameters(new_params)
        else:
            self.model.set_parameters_flat(new_params)
        # next_slice = 0
        # for p in self.model.parameters():
        #     slice_length = len(p.shape[-1])
        #     # with torch.no_grad():
        #         # p.copy_(new_params[next_slice: next_slice + slice_length].view(p.shape))
        #     p.copy(new_params[next_slice: next_slice + slice_length].reshape(p.shape))
        #     next_slice += slice_length

    def get_gradient(self):
        """
        Returns the gradients stacked into a 1d-tensor.
        """
        # TODO: reformulate for JAX
        gradients = [p.grad for p in self.model.parameters()]
        if None in gradients:
            return None
        return torch.cat([g.view(-1) for g in gradients]).view(-1)

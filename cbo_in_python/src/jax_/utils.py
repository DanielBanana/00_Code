import torch
import numpy as np
import jax.numpy as jnp
import jax.random as jrandom

def inplace_randn(size, device=None):
    device = torch.device('cpu') if device is None else device
    # https://discuss.pytorch.org/t/random-number-generation-speed/12209
    if device.type == 'cuda':
        return torch.cuda.FloatTensor(*size).normal_(0, 1)
    return torch.FloatTensor(*size).normal_(0, 1)

def randn(size, device=None):
    key = jrandom.PRNGKey(np.random.randint(0, 100))
    noise = jrandom.normal(key, size)
    return noise
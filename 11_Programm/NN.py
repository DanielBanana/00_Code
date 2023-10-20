import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'
import numpy as np

import jax
from jax import random, jit, flatten_util, numpy as jnp
from flax import linen as nn
from flax.core import freeze, unfreeze
import orbax.checkpoint
from flax.training import orbax_utils
from typing import Sequence
import sys
import time
import argparse
from functools import partial
from jax import lax
import warnings
import logging

# To use the plot_results file we need to add the uppermost folder to the PYTHONPATH
# Only Works if file gets called from 00_Code
sys.path.insert(0, os.getcwd())
from plot_results import plot_results, plot_losses, get_file_path
from utils import build_plot, result_plot_multi_dim, create_results_directory, create_results_subdirectories, create_doe_experiments, create_experiment_directory, visualise_wb
import yaml

# The Neural Network structure class
class ExplicitMLP(nn.Module):
    features: Sequence[int]
    def setup(self):
        self.layers = [nn.Dense(feat, kernel_init=nn.initializers.normal(0.0), bias_init=nn.initializers.normal(0.0)) for feat in self.features]

    # layers = []
    # for feat in self.features:
    #     layers.append(nn.Dense(feat))
    #     layers.append(nn.Dropout(0.2))
    # self.layers = layers

    def __call__(self, inputs):
        x = inputs
        for i, lyr in enumerate(self.layers):
            x = lyr(x)
            if i != len(self.layers) - 1:
                x = nn.relu(x)
        return x

def create_nn(layers, x0):
    key1, key2, = random.split(random.PRNGKey(np.random.randint(0,100)), 2)
    neural_network = ExplicitMLP(features=layers)
    parameters = neural_network.init(key2, np.zeros((1, x0.shape[0])))
    jitted_neural_network = jax.jit(lambda parameters, inputs: neural_network.apply(parameters, inputs))
    return jitted_neural_network, parameters, neural_network


def df_dtheta_function_FMU(df_dinput, dinput_dtheta):
    """Calculate the jacobian of the hybrid function (FMU + ML) with respect to the ML
    parameters

    Parameters
    ----------
    df_dinput : _type_
        _description_
    z : _type_
        _description_
    model_parameters : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    dinput_dtheta = unfreeze(dinput_dtheta)
    for layer in dinput_dtheta['params']:
        # Matrix multiplication inform of einstein sums (only really needed for the kernel
        # calculation, but makes the code uniform)
        dinput_dtheta['params'][layer]['bias'] = jnp.einsum("ij,jk->ik", df_dinput, dinput_dtheta['params'][layer]['bias'])
        dinput_dtheta['params'][layer]['kernel'] = jnp.einsum("ij,jkl->ikl", df_dinput, dinput_dtheta['params'][layer]['kernel'])
    # dinput_dtheta, should now be called df_dtheta
    return dinput_dtheta
vectorized_df_dtheta_function_FMU = jax.jit(jax.vmap(df_dtheta_function_FMU, in_axes=(0, 0)))
import jax
import numpy as np
import jax.numpy as jnp
import flax
from flax.core import unfreeze
from jax import random, numpy as jnp
from flax import linen as nn
from typing import Sequence
import os

from create_parameters_h import create_params_header
from create_cpp_code import create_cpp_code
from create_modelDescription import create_modelDescription
from build_fmu import build_fmu

def create_NN_FMU(targetDirPath, modelName, params, n_inputs, n_outputs):
    if os.path.exists(os.path.join(targetDirPath, "build", modelName, "modelDescription.xml")):
        print("NN at Path already created. Just the parameters will get replaced")
        create_params_header(targetDirPath=targetDirPath, params=params)
        build_fmu(targetDirPath=targetDirPath, modelName=modelName)
    else:
        guid = create_modelDescription(targetDirPath=targetDirPath, n_inputs=n_inputs, n_outputs=n_outputs)
        create_params_header(targetDirPath=targetDirPath, params=params)
        create_cpp_code(targetDirPath=targetDirPath,
                        modelName=modelName,
                        n_inputs=n_inputs,
                        n_outputs=n_outputs,
                        guid=guid)
        build_fmu(targetDirPath=targetDirPath, modelName=modelName)
import jax
import numpy as np
import jax.numpy as jnp
import flax
from flax.core import unfreeze
from jax import random, numpy as jnp
from flax import linen as nn
from typing import Sequence
import os

def extract_params(pytree):
    flat_params = flax.traverse_util.flatten_dict(pytree)
    flat_params = {k: jnp.array(v) for k, v in flat_params.items()}
    return flat_params

def create_params_header(targetDirPath, params):

    file_path = os.path.join(targetDirPath, "src", "weights_biases.h")

    flat_params = extract_params(params)

    # Generate C++ header file content
    cpp_header_content = '#ifndef WEIGHTS_BIASES_H\n'
    cpp_header_content += '#define WEIGHTS_BIASES_H\n\n'
    cpp_header_content += '#include <vector>\n\n'
    cpp_header_content += 'namespace NeuralNetworkParams {\n'

    layer_weights = {}  # Store layer weights as nested vectors
    layer_biases = {}   # Store layer biases as vectors
    layer_sizes = []

    for key, param in flat_params.items():
        layer_name = key[1]  # Extract the layer name
        param_name = key[-1]  # Extract the parameter name ('kernel' or 'bias')

        if param_name == 'kernel':
            if layer_name not in layer_weights:
                # Convert the parameter values to a nested vector of doubles
                param_values = param.tolist()
                layer_weights[layer_name] = param_values
                layer_sizes.append(str(len(layer_weights[layer_name])))
        elif param_name == 'bias':
            if layer_name not in layer_biases:
                # Convert the parameter values to a vector of doubles
                param_values = param.tolist()
                layer_biases[layer_name] = param_values
    layer_sizes.append(str(len(layer_weights[layer_name][0])))

    # Write weights
    cpp_header_content += '\tconst std::vector<std::vector<std::vector<double>>> weights = {\n'
    for layer_name, weights in layer_weights.items():
        cpp_header_content += '\t\t// ' + layer_name + ' weights\n'
        cpp_header_content += '\t\t{\n'
        for weight_matrix in weights:
            cpp_header_content += '\t\t\t{'
            cpp_header_content += ', '.join(map(str, weight_matrix))
            cpp_header_content += '},\n'
        cpp_header_content += '\t\t},\n'
    cpp_header_content += '\t};\n'

    # Write biases
    cpp_header_content += '\n\tconst std::vector<std::vector<double>> biases = {\n'
    for layer_name, biases in layer_biases.items():
        cpp_header_content += '\t\t// ' + layer_name + ' biases\n'
        cpp_header_content += '\t\t{'
        cpp_header_content += ', '.join(map(str, biases))
        cpp_header_content += '},\n'
    cpp_header_content += '\t};\n'

    layer_sizes = ', '.join(layer_sizes)

    cpp_header_content += '}\n'

    cpp_header_content += 'std::vector<int> layer_sizes = {{{LAYER_SIZES}}};\n'.format(LAYER_SIZES = layer_sizes)

    cpp_header_content += '\n#endif\n'


    # Write the C++ header file
    with open(file_path, 'w') as header_file:
        header_file.write(cpp_header_content)
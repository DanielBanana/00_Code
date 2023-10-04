import jax
import numpy as np
import jax.numpy as jnp
import flax
from flax.core import unfreeze
from jax import random, numpy as jnp
from flax import linen as nn
from typing import Sequence

# The Neural Network structure class
class ExplicitMLP(nn.Module):
    features: Sequence[int]

    def setup(self):
        # we automatically know what to do with lists, dicts of submodules
        self.layers = [nn.Dense(feat) for feat in self.features]
        # for single submodules, we would just write:
        # self.layer1 = nn.Dense(feat1)

    def __call__(self, inputs):
        x = inputs
        for i, lyr in enumerate(self.layers):
            x = lyr(x)
            if i != len(self.layers) - 1:
                x = nn.relu(x)
        return x

layers = [15, 1]
key1, key2 = random.split(random.PRNGKey(0), 2)
neural_network = ExplicitMLP(features=layers)
params = neural_network.init(key2, np.zeros((1, 2)))

# Replace this with your Pytree containing network parameters
# params = {}  # Example Pytree


def extract_params(pytree):
    flat_params = flax.traverse_util.flatten_dict(pytree)
    flat_params = {k: jnp.array(v) for k, v in flat_params.items()}
    return flat_params

flat_params = extract_params(params)

# Generate C++ header file content
cpp_header_content = '#ifndef WEIGHTS_BIASES_H\n'
cpp_header_content += '#define WEIGHTS_BIASES_H\n\n'
cpp_header_content += '#include <vector>\n\n'
cpp_header_content += 'namespace NeuralNetworkParams {\n'

layer_weights = {}  # Store layer weights as nested vectors
layer_biases = {}   # Store layer biases as vectors

for key, param in flat_params.items():
    layer_name = key[1]  # Extract the layer name
    param_name = key[-1]  # Extract the parameter name ('kernel' or 'bias')

    if param_name == 'kernel':
        if layer_name not in layer_weights:
            # Convert the parameter values to a nested vector of doubles
            param_values = param.tolist()
            layer_weights[layer_name] = param_values
    elif param_name == 'bias':
        if layer_name not in layer_biases:
            # Convert the parameter values to a vector of doubles
            param_values = param.tolist()
            layer_biases[layer_name] = param_values

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

cpp_header_content += '}\n\n#endif\n'

# Write the C++ header file
with open('weights_biases.h', 'w') as header_file:
    header_file.write(cpp_header_content)
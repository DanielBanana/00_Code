import jax
import flax.linen as nn
from jax import random
from typing import Sequence
import numpy as np

# The Neural Network structure class
class ExplicitMLP(nn.Module):
    features: Sequence[int]
    def setup(self):
        self.layers = [nn.Dense(feat, kernel_init=nn.initializers.normal(1.0), bias_init=nn.initializers.normal(1.0)) for feat in self.features]

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

def create_nn(layers, z0):
    key1, key2, = random.split(random.PRNGKey(np.random.randint(0,100)), 2)
    neural_network = ExplicitMLP(features=layers)
    nn_parameters = neural_network.init(key2, np.zeros((1, z0.shape[0])))
    jitted_neural_network = jax.jit(lambda p, x: neural_network.apply(p, x))
    return jitted_neural_network, nn_parameters

if __name__ == '__main__':
    z0 = np.zeros([1,0])
    layers = [15]
    layers.append(1)
    jitted_neural_network, nn_parameters = create_nn(layers, z0)
    print(nn_parameters)
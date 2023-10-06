import jax
import flax.linen as nn
from jax import random
from typing import Sequence
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import SymLogNorm
import os
from jax import flatten_util
from types import GeneratorType
from flax.core.frozen_dict import FrozenDict


# The Neural Network structure class
class ExplicitMLP(nn.Module):
    features: Sequence[int]
    def setup(self):
        self.layers = [nn.Dense(feat, kernel_init=nn.initializers.normal(1.0), bias_init=nn.initializers.normal(10.0)) for feat in self.features]

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

def visualise_wb(wb, plot_directory, name_addon):
    fig_all = plt.figure()
    ax_all = fig_all.subplots(2, len(wb['params'].keys()), sharex='col')

    fig_all.supxlabel('Layers')
    fig_all.supylabel('Biases   &   Weights')

    cmap = cm.get_cmap('viridis')

    if isinstance(wb, GeneratorType):
        # Parameters are from pytorch application
        pass
    if isinstance(wb, FrozenDict):
        pass
    flat, _ = flatten_util.ravel_pytree(wb)
    flat = np.array(flat)
    minimum = np.min(flat)
    maximum = np.max(flat)
    normalizer = SymLogNorm(linthresh=1.0, vmin=minimum, vmax=maximum)
    im = cm.ScalarMappable(cmap=cmap, norm=normalizer)

    if isinstance(wb, GeneratorType):
        for i, layer in enumerate(wb):
            # In the Pytorch parameter object first the weights and then the biases come
            matrix = layer.detach().numpy()

            fig = plt.figure()
            ax = fig.subplots(1,1)
            if len(matrix.shape) == 1:
                matrix = np.expand_dims(matrix,0)
                img = ax.imshow(matrix)
                img_all = ax_all[i,j].imshow(matrix,cmap=cmap,norm=normalizer)
            else:
                img = ax.imshow(matrix)
                img_all = ax_all[i,j].imshow(matrix,cmap=cmap,norm=normalizer)


            if matrix.shape[0] != 1:
                ax_all[i,j].set_yticks(range(0, matrix.shape[0], 5))
                ax.set_yticks(range(0, matrix.shape[0], 5))
            else:
                ax_all[i,j].set_yticks([])
                ax.set_yticks([])

            if matrix.shape[1] != 1:
                ax_all[i,j].set_xticks(range(0, matrix.shape[1], 5,))
                ax.set_xticks(range(0, matrix.shape[1], 5))
            else:
                ax_all[i,j].set_xticks([])
                ax.set_xticks([])

            fig.colorbar(img)
            if i%2 == 0:
                part = 'kernel'
            else:
                part = 'bias'
            fig.savefig(os.path.join(plot_directory, f'{name_addon}_layer_{int(i/2)}_{part}.png'))

    elif isinstance(wb, FrozenDict):
        for j, layer in enumerate(wb['params']):
            for i, part in enumerate(wb['params'][layer]):
                matrix = np.array(wb['params'][layer][part])
                print(wb['params'][layer][part])
                fig = plt.figure()
                ax = fig.subplots(1,1)
                if len(matrix.shape) == 1:
                    matrix = np.expand_dims(matrix,0)
                    img = ax.imshow(matrix)
                    img_all = ax_all[i,j].imshow(matrix,cmap=cmap,norm=normalizer)
                else:
                    img = ax.imshow(matrix)
                    img_all = ax_all[i,j].imshow(matrix,cmap=cmap,norm=normalizer)


                if matrix.shape[0] != 1:
                    ax_all[i,j].set_yticks(range(0, matrix.shape[0], 5))
                    ax.set_yticks(range(0, matrix.shape[0], 5))
                else:
                    ax_all[i,j].set_yticks([])
                    ax.set_yticks([])

                if matrix.shape[1] != 1:
                    ax_all[i,j].set_xticks(range(0, matrix.shape[1], 5,))
                    ax.set_xticks(range(0, matrix.shape[1], 5))
                else:
                    ax_all[i,j].set_xticks([])
                    ax.set_xticks([])

                fig.colorbar(img)
                fig.savefig(os.path.join(plot_directory, f'{name_addon}_{layer}_{part}.png'))

    fig_all.colorbar(im, ax=np.ravel(ax_all).tolist())
    fig_all.savefig(os.path.join(plot_directory, f'{name_addon}_all.png'))

if __name__ == '__main__':
    z0 = np.array([1,0])
    layers = [15, 20]
    layers.append(1)
    jitted_neural_network, nn_parameters = create_nn(layers, z0)

    path = os.path.abspath(__file__)
    directory = os.path.sep.join(path.split(os.path.sep)[:-1])

    visualise_wb(nn_parameters, directory, "values")
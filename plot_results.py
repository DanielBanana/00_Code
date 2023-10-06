import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import SymLogNorm
import os
from jax import flatten_util
from types import GeneratorType
from flax.core.frozen_dict import FrozenDict


def get_file_path(path):
    directory_of_this_file = '/'.join(path.split('/')[:-1])
    file_name_no_ext = os.path.basename(path).split('/')[-1].split('.')[0]
    plot_path = os.path.join(directory_of_this_file, file_name_no_ext)
    return plot_path

def plot_results(t, z, z_ref, path):
    fig = plt.figure()
    x_ax, v_ax = fig.subplots(2,1)
    x_ax.set_title('Position')
    x_ax.plot(t, z_ref[:,0], label='ref')
    v_ax.plot(t, z_ref[:,1], label='ref')
    v_ax.set_title('Velocity')
    if z is not None:
        x_ax.plot(t, z[:,0], label='sol')
        v_ax.plot(t, z[:,1], label='sol')
    x_ax.legend()
    v_ax.legend()
    fig.tight_layout()
    fig.savefig(f'{path}.png')
    plt.close(fig)

def plot_losses(epochs, training_losses, validation_losses=None, path=None):
    fig = plt.figure()
    ax = fig.subplots()
    ax.set_title('Loss')
    ax.plot(epochs, training_losses, label='Training')
    if validation_losses is not None:
        ax.plot(epochs, validation_losses, label='Validation')
        ax.legend()
    fig.tight_layout()
    if path is not None:
        fig.savefig(f'{path}.png')
    else:
        fig.savefig()
    plt.close(fig)

def visualise_wb(wb, plot_directory, name_addon):
    fig_all = plt.figure()
    fig_all.supxlabel('Layers')
    fig_all.supylabel('Biases   &   Weights')
    cmap = cm.get_cmap('viridis')

    if isinstance(wb, list):
        # Parameters are from pytorch application
        counter = 0
        minimum = np.inf
        maximum = -np.inf
        for i, layer in enumerate(wb):
            if layer.detach().numpy().max() > maximum:
                maximum = layer.detach().numpy().max()
            if layer.detach().numpy().min() < minimum:
                minimum = layer.detach().numpy().min()
            counter = i
        ax_all = fig_all.subplots(2, int((counter+1)/2))
    if isinstance(wb, FrozenDict):
        ax_all = fig_all.subplots(2, len(wb['params'].keys()), sharex='col')
        flat, _ = flatten_util.ravel_pytree(wb)
        flat = np.array(flat)
        minimum = np.min(flat)
        maximum = np.max(flat)
    normalizer = SymLogNorm(linthresh=1.0, vmin=minimum, vmax=maximum)
    im = cm.ScalarMappable(cmap=cmap, norm=normalizer)

    if isinstance(wb, list):
        for k, part in enumerate(wb):
            # In the Pytorch parameter object first the weights and then the biases come
            if k%2 == 0:
                part_name = 'kernel'
            else:
                part_name = 'bias'
            j = int(k/2)
            i = k%2

            matrix = part.detach().numpy()

            fig = plt.figure()
            ax = fig.subplots(1,1)
            if len(matrix.shape) == 1:
                matrix = np.expand_dims(matrix,0)

            if part_name == 'kernel':
                matrix = matrix.T

            img = ax.imshow(matrix,cmap=cmap,norm=normalizer)
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
            fig.savefig(os.path.join(plot_directory, f'layer_{j}_{part_name}_{name_addon}.png'))
            plt.close(fig)
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
                plt.close(fig)

    fig_all.colorbar(im, ax=np.ravel(ax_all).tolist())
    fig_all.savefig(os.path.join(plot_directory, f'all_{name_addon}.png'))
    plt.close(fig_all)


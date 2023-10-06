import datetime
import jax
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import SymLogNorm
import numpy as np
import os
from pyDOE2 import fullfact
from flax.core.frozen_dict import FrozenDict

def result_plot(model_name, dataset_name, plot_path,
                X_train, y_train, X_test, y_test, X_reference, y_reference):
    plt.rcParams['figure.figsize'] = (20, 10)
    plt.rcParams['font.size'] = 25

    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.plot(X_reference, y_reference, label='ref')
    ax1.scatter(X_train, y_train, label='train')
    ax1.legend()
    ax1.set_xlabel('X')
    ax1.set_ylabel('y')
    ax1.set_title('Train')

    ax2.plot(X_reference, y_reference, label='ref')
    ax2.scatter(X_test, y_test, label='test')
    ax2.legend()
    ax2.set_xlabel('X')
    ax2.set_ylabel('y')
    ax2.set_title('Test')

    plt.suptitle(f'{model_name} @ {dataset_name}')
    plt.savefig(plot_path)
    plt.close(fig)

def result_plot_multi_dim(model_name, dataset_name, plot_path,
        input_train, output_train, input_test, output_test, input_reference, output_reference, scatter=False):
    plt.rcParams['figure.figsize'] = (20, 10)
    plt.rcParams['font.size'] = 25

    # Get the dimensions of the output
    output_dims = output_train.shape[1]

    fig, axes = plt.subplots(output_dims, 2)

    for out_dim in range(output_dims):

        axes[out_dim, 0].plot(input_reference, output_reference[:,out_dim], label='Reference')
        if scatter:
            axes[out_dim, 0].scatter(input_train, output_train[:,out_dim], label='Prediction')
        else:
            axes[out_dim, 0].plot(input_train, output_train[:,out_dim], label='Prediction')
        axes[out_dim, 0].legend()
        axes[out_dim, 0].set_xlabel('X')
        axes[out_dim, 0].set_ylabel('y')
        axes[out_dim, 0].set_title(f'Variable {out_dim+1} - Train')

        axes[out_dim, 1].plot(input_reference, output_reference[:,out_dim], label='Reference')
        if scatter:
            axes[out_dim, 1].scatter(input_test, output_test[:,out_dim], label='Prediction')
        else:
            axes[out_dim, 1].plot(input_test, output_test[:,out_dim], label='Prediction')
        axes[out_dim, 1].legend()
        axes[out_dim, 1].set_xlabel('X')
        axes[out_dim, 1].set_ylabel('y')
        axes[out_dim, 1].set_title(f'Variable {out_dim+1} - Test')

    fig.tight_layout()
    fig.suptitle(f'{model_name} @ {dataset_name}')
    fig.savefig(plot_path)
    plt.close(fig)

def build_plot(epochs, model_name, dataset_name, plot_path,
               train_acc, test_acc, train_loss, test_loss):
    plt.rcParams['figure.figsize'] = (20, 10)
    plt.rcParams['font.size'] = 25

    epochs_range = np.arange(1, epochs + 1, dtype=int)

    plt.clf()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    ax1.plot(epochs_range, train_acc, label='train')
    ax1.plot(epochs_range, test_acc, label='test')
    ax1.legend()
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('accuracy')
    ax1.set_title('Accuracy')

    ax2.plot(epochs_range, train_loss, label='train')
    ax2.plot(epochs_range, test_loss, label='test')
    ax2.legend()
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('loss')
    ax2.set_title('Loss')

    ax3.plot(epochs_range, train_loss, label='train')
    ax3.plot(epochs_range, test_loss, label='test')
    ax3.set_yscale('log')
    ax3.legend()
    ax3.set_xlabel('epoch')
    ax3.set_ylabel('loss')
    ax3.set_title('Loss')

    plt.suptitle(f'{model_name} @ {dataset_name}')
    plt.savefig(plot_path)
    plt.close(fig)

def visualise_wb(wb, plot_directory, name_addon, plot_individual=False):
    fig_all = plt.figure()
    fig_all.supxlabel('Layers')
    fig_all.supylabel('Biases   &   Weights')
    cmap = cm.get_cmap('seismic')

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
    elif isinstance(wb, FrozenDict):
        # Parameters are from FLAX/JAX
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
            if len(matrix.shape) == 1:
                matrix = np.expand_dims(matrix,0)
            if part_name == 'kernel':
                matrix = matrix.T

            if plot_individual:
                fig = plt.figure()
                ax = fig.subplots(1,1)
                img = ax.imshow(matrix,cmap=cmap,norm=normalizer)

            img_all = ax_all[i,j].imshow(matrix,cmap=cmap,norm=normalizer)

            if matrix.shape[0] != 1:
                ax_all[i,j].set_yticks(range(0, matrix.shape[0], 5))
                if plot_individual:
                    ax.set_yticks(range(0, matrix.shape[0], 5))
            else:
                ax_all[i,j].set_yticks([])
                if plot_individual:
                    ax.set_yticks([])

            if matrix.shape[1] != 1:
                ax_all[i,j].set_xticks(range(0, matrix.shape[1], 5,))
                if plot_individual:
                    ax.set_xticks(range(0, matrix.shape[1], 5))
            else:
                ax_all[i,j].set_xticks([])
                if plot_individual:
                    ax.set_xticks([])

            if plot_individual:
                fig.colorbar(img)
                fig.savefig(os.path.join(plot_directory, f'layer_{j}_{part_name}_{name_addon}.png'))
                plt.close(fig)
    elif isinstance(wb, FrozenDict):
        for j, layer in enumerate(wb['params']):
            for i, part in enumerate(wb['params'][layer]):
                matrix = np.array(wb['params'][layer][part])
                if len(matrix.shape) == 1:
                    matrix = np.expand_dims(matrix,0)
                # print(wb['params'][layer][part])
                if plot_individual:
                    fig = plt.figure()
                    ax = fig.subplots(1,1)
                    img = ax.imshow(matrix)

                img_all = ax_all[i,j].imshow(matrix,cmap=cmap,norm=normalizer)

                if matrix.shape[0] != 1:
                    ax_all[i,j].set_yticks(range(0, matrix.shape[0], 5))
                    if plot_individual:
                        ax.set_yticks(range(0, matrix.shape[0], 5))
                else:
                    ax_all[i,j].set_yticks([])
                    if plot_individual:
                        ax.set_yticks([])

                if matrix.shape[1] != 1:
                    ax_all[i,j].set_xticks(range(0, matrix.shape[1], 5,))
                    if plot_individual:
                        ax.set_xticks(range(0, matrix.shape[1], 5))
                else:
                    ax_all[i,j].set_xticks([])
                    if plot_individual:
                        ax.set_xticks([])

                if plot_individual:
                    fig.colorbar(img)
                    fig.savefig(os.path.join(plot_directory, f'{name_addon}_{layer}_{part}.png'))
                    plt.close(fig)

    fig_all.colorbar(im, ax=np.ravel(ax_all).tolist())
    fig_all.savefig(os.path.join(plot_directory, f'all_{name_addon}.png'))
    plt.close(fig_all)

def create_results_directory(directory, results_directory_name=None):
    if results_directory_name is None:
        now = datetime.datetime.now()
        date = '-'.join([str(now.year), str(now.month), str(now.day)]) + '_' + '-'.join([str(now.hour), str(now.minute)])
        results_directory = os.path.join(directory, date)
    else:
        results_directory = os.path.join(directory, results_directory_name)
        if not os.path.exists(results_directory):
            os.mkdir(results_directory)
        else:
            count = 1
            while os.path.exists(results_directory):
                results_directory = os.path.join(directory, results_directory_name + f'_{count}')
                count += 1
            os.mkdir(results_directory)
    return results_directory

def create_results_subdirectories(results_directory, trajectory=False, residual=False, checkpoint=True):
    return_directories = []
    if trajectory:
        trajectory_directory = os.path.join(results_directory, 'trajectory')
        if not os.path.exists(trajectory_directory):
            os.mkdir(trajectory_directory)
        return_directories.append(trajectory_directory)

    if residual:
        residual_directory = os.path.join(results_directory, 'residual')
        if not os.path.exists(residual_directory):
            os.mkdir(residual_directory)
        return_directories.append(residual_directory)

    if checkpoint:
        checkpoint_directory = os.path.join(results_directory, 'ckpt')
        if not os.path.exists(checkpoint_directory):
            os.mkdir(checkpoint_directory)
        return_directories.append(checkpoint_directory)

    return tuple(return_directories)

def create_experiment_directory(doe_directory, n_exp):
    experiment_directory = os.path.join(doe_directory, f'Experiment {n_exp}')
    if not os.path.exists(experiment_directory):
        os.mkdir(experiment_directory)
    return experiment_directory

def create_doe_experiments(doe_parameters, method='fullfact'):
    levels = [len(val) for val in doe_parameters.values()]
    if method == 'fullfact':
        doe = fullfact(levels)
    else:
        print('Method not supported, using fullfact')
        doe = fullfact(levels)
    experiments = []
    for experiment in doe:
        experiment_dict = {}
        for i, key in enumerate(doe_parameters.keys()):
            experiment_dict[key] = doe_parameters[key][int(experiment[i])]
        experiments.append(experiment_dict)
    return tuple(experiments)

def create_residual_references(z_ref, t, variables, ode_res):
    z_dot = (z_ref[1:] - z_ref[:-1])/(t[1:] - t[:-1]).reshape(-1,1)
    v_ode = jax.vmap(lambda z_ref, t, ode_parameters: ode_res(z_ref, t, ode_parameters), in_axes=(0, 0, None))
    residual = z_dot - v_ode(z_ref[:-1], t[:-1], variables)
    return residual

def create_residual_reference_solution(t_train, z_ref_train, t_test, z_ref_test, variables, ode_res):

    # CREATE RESIDUALS FROM TRAJECTORIES
    train_residual_outputs = np.asarray(create_residual_references(z_ref_train, t_train, variables, ode_res))[:,1]
    train_residual_outputs = train_residual_outputs.reshape(-1, 1) # We prefer it if the output has a two dimensional shape (n_samples, output_dim) even if the output_dim is 1
    train_residual_inputs = z_ref_train[:-1]

    test_residual_outputs = np.asarray(create_residual_references(z_ref_test, t_test, variables, ode_res))[:,1]
    test_residual_outputs = test_residual_outputs.reshape(-1, 1)
    test_residual_inputs = z_ref_test[:-1]

    return train_residual_inputs, train_residual_outputs, test_residual_inputs, test_residual_outputs
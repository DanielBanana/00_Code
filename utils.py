import matplotlib.pyplot as plt
import numpy as np
import os
import datetime
from pyDOE2 import fullfact

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
import os
import sys
import argparse
import time
import warnings

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import datetime
sys.path.insert(1, os.getcwd())
from plot_results import plot_results, plot_losses, get_file_path

# sys.path.append(os.path.join(os.getcwd().split('cbo_in-python')[0], 'cbo-in-python'))

from cbo_in_python.src.torch.models import *
from cbo_in_python.src.datasets import load_mnist_dataloaders, load_parabola_dataloaders, load_shifted_parabola_dataloaders, get_mnist_dataset, load_generic_dataloaders
from cbo_in_python.src.torch.optimizer import Optimizer
from cbo_in_python.src.torch.loss import Loss
from torch.utils.data import Dataset, DataLoader

from collections import OrderedDict
from pyDOE2 import fullfact

from utils import build_plot, result_plot, create_results_directory, create_doe_experiments, create_experiment_directory

MODELS = {
    'MNIST_726x10': MNIST_726x10,
    'MNIST_726x20': MNIST_726x20,
    'MNIST_726x10x10': MNIST_726x10x10,
    'PARA_5x5x5' : PARA_5x5x5,
    'PARA_7x7': PARA_7x7,
    'PARA_25': PARA_25,
    'PARA_15': PARA_15,
    'PARA_5': PARA_5,
    'PARA_2x25': PARA_2x25,
    'LeNet1': LeNet1,
    'LeNet5': LeNet5,
    'Net': Net
}

DATASETS = {
    'MNIST': load_mnist_dataloaders,
    'PARABOLA': load_parabola_dataloaders,
    'SHIFTED-PARABOLA': load_shifted_parabola_dataloaders
}

def _evaluate_class(model, X_, y_, loss_fn):
    with torch.no_grad():
        outputs = model(X_)
        y_pred = torch.argmax(outputs, dim=1)
        loss = loss_fn(outputs, y_)
        acc = 1. * y_.eq(y_pred).sum().item() / y_.shape[0]
    return loss, acc

def _evaluate_reg(model, X_, y_, loss_fn):
    with torch.no_grad():
        outputs = model(X_)
        loss = loss_fn(outputs, y_)
    return loss

def number_of_nn_evaluations(n_train_batches,
                             n_test_batches,
                             n_particles,
                             particle_batch_size,
                             epochs):
    n = n_particles/particle_batch_size

    return int(((1+n)*n_train_batches + n_test_batches)*epochs)

def train(model, train_dataloader, test_dataloader, device, use_multiprocessing, processes,
          epochs, particles, particles_batch_size,
          alpha, sigma, l, dt, anisotropic, eps, partial_update, cooling,
          log_interval, problem_type, run_file):
    train_accuracies = []
    train_losses = []
    test_accuracies = []
    test_losses = []

    optimizer = Optimizer(model, n_particles=particles, alpha=alpha, sigma=sigma,
                          l=l, dt=dt, anisotropic=anisotropic, eps=eps, partial_update=partial_update,
                          use_multiprocessing=use_multiprocessing, n_processes=processes,
                          particles_batch_size=particles_batch_size, device=device)

    if problem_type == 'classification':
        loss_fn = Loss(F.nll_loss, optimizer)
    else:
        loss_fn = Loss(F.mse_loss, optimizer)

    n_batches = len(train_dataloader)

    for epoch in range(epochs):
        epoch_train_accuracies = []
        epoch_train_losses = []
        for batch_idx, (data, y) in enumerate(train_dataloader):
            data, y = data.to(device), y.to(device)
            if problem_type == 'classification':
                loss_train, acc_train = _evaluate_class(model, data, y, F.nll_loss)
            else:
                loss_train = _evaluate_reg(model, data, y, F.mse_loss)
                acc_train = 0.0

            optimizer.zero_grad()
            loss_fn.backward(data, y, backward_gradients=False)
            optimizer.step()

            epoch_train_accuracies.append(acc_train)
            epoch_train_losses.append(loss_train.cpu().item())

            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_dataloader.dataset),
                    100. * batch_idx / len(train_dataloader), loss_train.item()))


        train_accuracies.append(np.mean(epoch_train_accuracies))
        train_losses.append(np.mean(epoch_train_losses))
        with open(run_file, 'a') as file:
            file.write('\nTrain Epoch: {} \tLoss: {:.6f}'.format(epoch, np.mean(epoch_train_losses)))

        with torch.no_grad():
            losses = []
            accuracies = []
            for X_test, y_test in test_dataloader:
                X_test, y_test = X_test.to(device), y_test.to(device)
                if problem_type == 'classification':
                    loss_test, acc_test = _evaluate_class(model, X_test, y_test, F.nll_loss)
                else:
                    loss_test = _evaluate_reg(model, X_test, y_test, F.mse_loss)
                    acc_test = 0.0
                losses.append(loss_test.cpu().item())
                accuracies.append(acc_test)
            loss_test, acc_test = np.mean(losses), np.mean(accuracies)
            test_losses.append(loss_test)
            test_accuracies.append(acc_test)
        print('\nTest set: Average loss: {:.4f}, Accuracy: ({:.0f}%)\n'.format(loss_test, acc_test*100))
        with open(run_file, 'a') as file:
            file.write('\nTest set: Average Loss: {:.4f}, Accuracy: ({:.0f}%)\n'.format(loss_test, acc_test*100))

            # print(
            #     f'Epoch: {epoch + 1:2}/{epochs}, batch: {batch + 1:4}/{n_batches}, train loss: {train_loss:8.3f}, '
            #     f'train acc: {train_acc:8.3f}, test loss: {val_loss:8.3f}, test acc: {val_acc:8.3f}, alpha: {optimizer.alpha:8.3f}, sigma: {optimizer.sigma:8.3f}',
            #     flush=True)

        if cooling:
            optimizer.cooling_step()

    return train_accuracies, test_accuracies, train_losses, test_losses

# def build_plot(epochs, model_name, dataset_name, plot_path,
#                train_acc, test_acc, train_loss, test_loss):
#     plt.rcParams['figure.figsize'] = (20, 10)
#     plt.rcParams['font.size'] = 25

#     epochs_range = np.arange(1, epochs + 1, dtype=int)

#     plt.clf()
#     fig, (ax1, ax2) = plt.subplots(1, 2)

#     ax1.plot(epochs_range, train_acc, label='train')
#     ax1.plot(epochs_range, test_acc, label='test')
#     ax1.legend()
#     ax1.set_xlabel('epoch')
#     ax1.set_ylabel('accuracy')
#     ax1.set_title('Accuracy')

#     ax2.plot(epochs_range, train_loss, label='train')
#     ax2.plot(epochs_range, test_loss, label='test')
#     ax2.legend()
#     ax2.set_xlabel('epoch')
#     ax2.set_ylabel('loss')
#     ax2.set_title('Loss')

#     plt.suptitle(f'{model_name} @ {dataset_name}')
#     plt.savefig(plot_path)

# def result_plot(model_name, dataset_name, plot_path,
#                 X_train, y_train, X_test, y_test, X_reference, y_reference):
#     plt.rcParams['figure.figsize'] = (20, 10)
#     plt.rcParams['font.size'] = 25

#     fig, (ax1, ax2) = plt.subplots(1, 2)

#     ax1.plot(X_reference, y_reference, label='ref')
#     ax1.scatter(X_train, y_train, label='train')
#     ax1.legend()
#     ax1.set_xlabel('X')
#     ax1.set_ylabel('y')
#     ax1.set_title('Train')

#     ax2.plot(X_reference, y_reference, label='ref')
#     ax2.scatter(X_test, y_test, label='test')
#     ax2.legend()
#     ax2.set_xlabel('X')
#     ax2.set_ylabel('y')
#     ax2.set_title('Test')

#     plt.suptitle(f'{model_name} @ {dataset_name}')
#     plt.savefig(plot_path)

# def create_results_directory(directory, results_directory_name=None):
#     if results_directory_name is None:
#         now = datetime.datetime.now()
#         doe_date = '-'.join([str(now.year), str(now.month), str(now.day)]) + '_' + '-'.join([str(now.hour), str(now.minute)])
#         doe_directory = os.path.join(directory, doe_date)
#     else:
#         doe_directory = os.path.join(directory, results_directory_name)
#         if not os.path.exists(doe_directory):
#             os.mkdir(doe_directory)
#         else:
#             count = 1
#             while os.path.exists(doe_directory):
#                 doe_directory = os.path.join(directory, results_directory_name + f'_{count}')
#                 count += 1
#             os.mkdir(doe_directory)
#     return doe_directory

# def create_doe_experiments(doe_parameters, method='fullfact'):
#     levels = [len(val) for val in doe_parameters.values()]
#     if method == 'fullfact':
#         doe = fullfact(levels)
#     else:
#         print('Method not supported, using fullfact')
#         doe = fullfact(levels)
#     experiments = []
#     for experiment in doe:
#         experiment_dict = {}
#         for i, key in enumerate(doe_parameters.keys()):
#             experiment_dict[key] = doe_parameters[key][int(experiment[i])]
#         experiments.append(experiment_dict)
#     return tuple(experiments)

# def create_experiment_directory(doe_directory, n_exp):
#     experiment_directory = os.path.join(doe_directory, f'Experiment {n_exp}')
#     if not os.path.exists(experiment_directory):
#         os.mkdir(experiment_directory)
#     return experiment_directory

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--model', type=str, default='SimpleMLP', help=f'architecture to use',
                        choices=list(MODELS.keys()))
    parser.add_argument('--dataset', type=str, default='MNIST', help='dataset to use',
                        choices=list(DATASETS.keys()))

    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='cuda',
                        help='whether to use GPU (cuda) for accelerated computations or not')
    parser.add_argument('--use_multiprocessing', action='store_true',
                        help='specify to use multiprocessing for accelerating computations on CPU '
                             '(note, it is impossible to use multiprocessing with GPU)')
    parser.add_argument('--processes', type=int, default=4,
                        help='how many processes to use for multiprocessing')

    parser.add_argument('--epochs', type=int, default=100, help='train for EPOCHS epochs')
    parser.add_argument('--batch_size', type=int, default=60, help='batch size (for samples-level batching)')
    parser.add_argument('--particles', type=int, default=100, help='')
    parser.add_argument('--particles_batch_size', type=int, default=20, help='batch size '
                                                                             '(for particles-level batching)')

    parser.add_argument('--alpha', type=float, default=50, help='alpha from CBO dynamics')
    parser.add_argument('--sigma', type=float, default=0.4 ** 0.5, help='sigma from CBO dynamics')
    parser.add_argument('--l', type=float, default=1, help='lambda from CBO dynamics')
    parser.add_argument('--dt', type=float, default=0.1, help='dt from CBO dynamics')
    parser.add_argument('--anisotropic', type=bool, default=True, help='whether to use anisotropic or not')
    parser.add_argument('--eps', type=float, default=1e-5, help='threshold for additional random shift')
    parser.add_argument('--partial_update', type=bool, default=False, help='whether to use partial or full update')
    parser.add_argument('--cooling', type=bool, default=False, help='whether to apply cooling strategy')

    parser.add_argument('--build_plot', required=False, action='store_true',
                        help='specify to build loss and accuracy plot')
    parser.add_argument('--plot_path', required=False, type=str, default='cbo.png',
                        help='path to save the resulting plot')

    parser.add_argument('--eval_freq', type=int, default=10, help='evaluate test accuracy every EVAL_FREQ '
                                                                   'samples-level batches')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--results_directory_name', required=False, type=str, default='CBO_MNIST_TEST',
                        help='name under which the results should be saved, like plots and such')
    parser.add_argument('--n_runs', type=int, default=10,
                        help='DoE Parameter; how often each configuration should be run to compute an average')

    doe = True
    compiled = False

    args = parser.parse_args()
    args.build_plot=True
    warnings.filterwarnings('ignore')

    if compiled:
        directory = os.getcwd()
    else:
        path = os.path.abspath(__file__)
        directory = os.path.sep.join(path.split(os.path.sep)[:-1])
        file_path = get_file_path(path)

    results_directory = create_results_directory(directory=directory, results_directory_name=args.results_directory_name)

    train_dataloader, test_dataloader = DATASETS[args.dataset](train_batch_size=args.batch_size,
                                                               test_batch_size=10000)

    print(results_directory)

    if args.dataset == 'PARABOLA' or args.dataset == 'SHIFTED-PARABOLA':
        problem_type = 'regression'
    elif args.dataset == 'MNIST':
        problem_type = 'classification'
    else:
        problem_type = 'classification'

    device = args.device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print('Cuda is unavailable. Using CPU instead.')
        device = 'cpu'
    use_multiprocessing = args.use_multiprocessing
    if device != 'cpu' and use_multiprocessing:
        print('Unable to use multiprocessing on GPU')
        use_multiprocessing = False
    device = torch.device(device)

    if doe:
        results_name =  'results.txt'
        setup_file_name = 'setup.yaml'
        plot_file_name = 'doe_results'

        results_file = os.path.join(results_directory, results_name)
        setup_file = os.path.join(results_directory, setup_file_name)
        plot_file = os.path.join(results_directory, plot_file_name)

        if args.dataset == 'PARABOLA' or args.dataset == 'SHIFTED-PARABOLA':
            if args.dataset == 'PARABOLA':
                doe_models = ['PARA_25', 'PARA_15', 'PARA_5']
            else:
                doe_models = ['PARA_2x25']
            doe_epochs = [15]
            doe_particles = [10, 100, 200]
        else:
            doe_models = ['MNIST_726x10', 'MNIST_726x10x10', 'MNIST_726x20']
            doe_epochs = [15]
            doe_particles = [10, 100, 200]
            # doe_epochs = [1, 2]
            # doe_particles = [4, 5]

        doe_parameters = OrderedDict({'models': doe_models,
                                      'particles': doe_particles,
                                      'epochs': doe_epochs})

        experiments = create_doe_experiments(doe_parameters, method='fullfact')

        experiment_result_by_epochs = {}
        for doe_epoch in doe_epochs:
            experiment_result_by_epochs['{}'.format(doe_epoch)] = {}

        for averaging_run in range(1, args.n_runs+1):

            for n_exp, experiment in enumerate(experiments):
                nn_model = experiment['models']
                particles = experiment['particles']
                epochs = experiment['epochs']
                model = MODELS[nn_model]()
                print(f'Training {nn_model} @ {args.dataset} (Run {averaging_run})')
                nn_evals = number_of_nn_evaluations(len(train_dataloader),
                                                    len(test_dataloader),
                                                    particles,
                                                    args.particles_batch_size,
                                                    epochs)
                trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

                experiment_directory = create_experiment_directory(results_directory, n_exp)
                run_file = os.path.join(experiment_directory, 'results.txt')
                parameter_file_name = '{}_{}.pt'.format(args.dataset, nn_model)
                parameter_file = os.path.join(experiment_directory, parameter_file_name)
                with open(run_file, 'a') as file:
                    file.write(str(experiment)+'\n')
                    file.write('Particles to Trainable Parameters: {}/{} ({:.3f}%)'.format(particles, trainable_parameters, particles/trainable_parameters*100))

                start_time = time.time()
                result = train(model, train_dataloader, test_dataloader, device, use_multiprocessing, args.processes,
                            epochs, particles, args.particles_batch_size,
                            args.alpha, args.sigma, args.l, args.dt, args.anisotropic, args.eps, args.partial_update,
                            args.cooling,
                            args.eval_freq,
                            problem_type=problem_type,
                            run_file=run_file)
                elapsed_time = time.time() - start_time
                accuracies_train, accuracies_test, losses_train, losses_test = result

                plot_dataloader_train, plot_dataloader_test = DATASETS[args.dataset](train_batch_size=60000,
                                                               test_batch_size=10000)

                # Should only be one iteration since batch_size = number of samples)
                # for X, y in (plot_dataloader_train):
                #     nn_output = model(X).detach()
                #     fig, ax = plt.subplots()
                #     ax.scatter(X, y, label='Reference')
                #     ax.scatter(X, nn_output, label='Prediction')
                #     ax.legend()
                #     fig.savefig(os.path.join(experiment_directory, 'prediction_' + args.plot_path))

                print('Elapsed time: {:.1f} seconds'.format(elapsed_time))
                if args.build_plot:
                    build_plot(epochs, nn_model, args.dataset, os.path.join(experiment_directory, 'loss_' + args.plot_path), *result)

                best_epoch = np.argmin(np.array(losses_test))
                best_accuracy_test = accuracies_test[best_epoch]
                best_loss_test = losses_test[best_epoch]

                if nn_model not in experiment_result_by_epochs['{}'.format(epochs)].keys():
                    experiment_result_by_epochs['{}'.format(epochs)][nn_model+f'_{particles}']={
                        'loss': [],
                        'acc': [],
                        'forward_evals': [],
                        'time': []
                    }

                # experiment_result_by_epochs['{}'.format(epochs)][nn_model]['particles'].append(particles)
                experiment_result_by_epochs['{}'.format(epochs)][nn_model+f'_{particles}']['loss'].append(losses_test)
                experiment_result_by_epochs['{}'.format(epochs)][nn_model+f'_{particles}']['acc'].append(accuracies_test)
                experiment_result_by_epochs['{}'.format(epochs)][nn_model+f'_{particles}']['forward_evals'].append(nn_evals)
                experiment_result_by_epochs['{}'.format(epochs)][nn_model+f'_{particles}']['time'].append(elapsed_time)


                # experiment_result_by_epochs['{}'.format(epochs)].append({'NN': nn_model,
                #                                                         'particles': particles,
                #                                                         'loss': losses_test,
                #                                                         'acc': accuracies_test,
                #                                                         'forward_evals': nn_evals,
                #                                                         'time': elapsed_time})
                with open(results_file, 'a') as file:
                    file.write('''\nExperiment {}/{}, Best Epoch: {}/{}, Best Loss: {:.4f}, Best Accuracy: ({:.0f}%), Final Loss: {:.4f}, Final Accuracy: ({:.0f}%), NN-Evaluations: {}, Part./Par.: {:.4f}%, Time: {:.1f}'''.format(
                        n_exp, len(experiments), best_epoch, epochs-1, best_loss_test,
                        best_accuracy_test*100, losses_test[-1], accuracies_test[-1]*100, nn_evals,
                        particles/trainable_parameters*100, elapsed_time
                    ))
                if args.save_model:
                    torch.save(model.state_dict(), parameter_file)

                # PLOTTIN THE RESULTS FOR ALL EXPERIMENTS IN 2 PLOTS
                for epoch, experiment_results in experiment_result_by_epochs.items():
                    title = 'Results for {} Epochs ({} runs)'.format(epoch, averaging_run)
                    plt.rcParams['figure.figsize'] = (20, 10)
                    plt.rcParams['font.size'] = 25
                    epochs_range = np.arange(1, int(epoch) + 1, dtype=int)
                    plt.clf()
                    fig, (ax1, ax2) = plt.subplots(1, 2)

                    for experiment_model, experiment in experiment_results.items():
                        loss_history = np.asarray(experiment['loss']).mean(axis=0)
                        acc_history = np.asarray(experiment['acc']).mean(axis=0)
                        label = '{}'.format(experiment_model)

                        ax1.plot(epochs_range, acc_history, label=label)
                        ax1.legend()
                        ax1.set_xlabel('epoch')
                        ax1.set_ylabel('accuracy')
                        ax1.set_title('Accuracy')

                        ax2.plot(epochs_range, loss_history, label=label)
                        ax2.legend()
                        ax2.set_xlabel('epoch')
                        ax2.set_ylabel('loss')
                        ax2.set_title('Loss')
                    plt.suptitle(title)
                    plt.savefig(plot_file + '_{}_Epochs.png'.format(epoch))

                # PLOTTING THE RESULTS FOR ALL EXPERIMENTS; WEIGHT THE PERFORMANCE BY NUMBER OF #PARTICLES
                for epoch, experiment_results in experiment_result_by_epochs.items():
                    title = 'Results for {} Epochs ({} runs)'.format(epoch, averaging_run)
                    plt.rcParams['figure.figsize'] = (20, 10)
                    plt.rcParams['font.size'] = 25
                    epochs_range = np.arange(1, int(epoch) + 1, dtype=int)
                    plt.clf()
                    fig, (ax1, ax2) = plt.subplots(1, 2)

                    for experiment_model, experiment in experiment_results.items():
                        # particles = experiment['particles'][0]
                        loss_history = np.asarray(experiment['loss']).mean(axis=0)
                        acc_history = np.asarray(experiment['acc']).mean(axis=0)
                        label = '{}'.format(experiment_model)
                        particles = int(experiment_model.split('_')[-1])

                        ax1.plot(epochs_range, np.asarray(acc_history)/particles, label=label)
                        ax1.legend()
                        ax1.set_xlabel('Epoch')
                        ax1.set_ylabel('Accuracy')
                        ax1.set_title('Accuracy by #Particles')

                        ax2.plot(epochs_range, np.asarray(loss_history)*particles, label=label)
                        ax2.legend()
                        ax2.set_xlabel('Epoch')
                        ax2.set_ylabel('Loss')
                        ax2.set_title('#Particles weighted Loss')
                    plt.suptitle(title)
                    plt.savefig(plot_file + '_{}_Epochs_weighted_by_particles.png'.format(epoch))

                # PLOTTING THE RESULTS FOR ALL EXPERIMENTS; WEIGHT THE PERFORMANCE BY NUMBER OF #PARAMETERS
                for epoch, experiment_results in experiment_result_by_epochs.items():
                    title = 'Results for {} Epochs ({} runs)'.format(epoch, averaging_run)
                    plt.rcParams['figure.figsize'] = (20, 10)
                    plt.rcParams['font.size'] = 25
                    epochs_range = np.arange(1, int(epoch) + 1, dtype=int)
                    plt.clf()
                    fig, (ax1, ax2) = plt.subplots(1, 2)

                    for experiment_model, experiment in experiment_results.items():
                        # particles = experiment['particles'][0]
                        loss_history = np.asarray(experiment['loss']).mean(axis=0)
                        acc_history = np.asarray(experiment['acc']).mean(axis=0)
                        label = '{}'.format(experiment_model)

                        ax1.plot(epochs_range, np.asarray(acc_history)/trainable_parameters, label=label)
                        ax1.legend()
                        ax1.set_xlabel('Epoch')
                        ax1.set_ylabel('Accuracy')
                        ax1.set_title('Accuracy by #Parameters')

                        ax2.plot(epochs_range, np.asarray(loss_history)*trainable_parameters, label=label)
                        ax2.legend()
                        ax2.set_xlabel('Epoch')
                        ax2.set_ylabel('Loss')
                        ax2.set_title('Parameters weighted Loss')
                    plt.suptitle(title)
                    plt.savefig(plot_file + '_{}_Epochs_weighted_by_parameters.png'.format(epoch))

                # Plot the Number of evaluations
                exp_idx = 0
                indices = []
                forward_evals = []
                for epoch, experiment_results in experiment_result_by_epochs.items():
                    for experiment_model, experiment in experiment_results.items():
                        indices.append(exp_idx)
                        forward_evals.append(np.asarray(experiment['forward_evals']).mean())
                        exp_idx += 1
                evals_fig, evals_ax = plt.subplots()
                evals_ax.bar(indices, forward_evals)
                # ax1.legend()
                evals_ax.set_xlabel('Experiment #')
                evals_ax.set_ylabel('Evaluations')
                evals_ax.set_title('Forward Evaluations ({} runs)'.format(averaging_run))
                evals_ax.yaxis.get_major_locator().set_params(integer=True)
                plt.savefig(plot_file + '_Evaluations.png')

                # Plot the time
                exp_idx = 0
                indices = []
                times = []
                for epoch, experiment_results in experiment_result_by_epochs.items():
                    for experiment_model, experiment in experiment_results.items():
                        indices.append(exp_idx)
                        times.append(np.asarray(experiment['time']).mean())
                        exp_idx += 1
                time_fig, time_ax = plt.subplots()
                time_ax.bar(indices, times)
                # ax1.legend()
                time_ax.set_xlabel('Experiment #')
                time_ax.set_ylabel('Time (s)')
                time_ax.set_title('Time ({} runs)'.format(averaging_run))
                time_ax.yaxis.get_major_locator().set_params(integer=True)
                time_fig.savefig(plot_file + '_Time.png')

    else:
        model = MODELS[args.model]()
        print(f'Training {args.model} @ {args.dataset}')
        nn_evals = number_of_nn_evaluations(len(train_dataloader),
                                            len(test_dataloader),
                                            args.particles,
                                            args.particles_batch_size,
                                            args.epochs)
        print(f'Number of total NN evaluations: {nn_evals}')
        start_time = time.time()
        result = train(model, train_dataloader, test_dataloader, device, use_multiprocessing, args.processes,
                    args.epochs, args.particles, args.particles_batch_size,
                    args.alpha, args.sigma, args.l, args.dt, args.anisotropic, args.eps, args.partial_update,
                    args.cooling,
                    args.eval_freq,
                    problem_type=problem_type)
        print(f'Elapsed time: {time.time() - start_time} seconds')
        if args.build_plot:
            build_plot(args.epochs, args.model, args.dataset, os.path.join(results_directory, 'loss_' + args.plot_path),
                    *result)
        if args.save_model:
            torch.save(model.state_dict(), "mnist_cnn.pt")

            # result_plot(args.model, args.dataset, 'predictions_' + args.plot_path, train_dataloader.dataset.x,
            #             model(train_dataloader.dataset.x).detach().numpy(), test_dataloader.dataset.x,
            #             model(test_dataloader.dataset.x).detach().numpy(), train_dataloader.dataset.x+test_dataloader.dataset.x,
            #             train_dataloader.dataset.y+test_dataloader.dataset.y)





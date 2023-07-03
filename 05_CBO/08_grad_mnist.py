import os
import sys
import argparse
import time
import warnings

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

sys.path.insert(1, os.getcwd())
from plot_results import plot_results, plot_losses, get_file_path
import datetime

# sys.path.append(os.path.join(os.getcwd().split('cbo-in-python')[0], 'cbo-in-python'))

from cbo_in_python.src.torch.models import *
from cbo_in_python.src.datasets import load_mnist_dataloaders, load_parabola_dataloaders, get_mnist_dataset, load_generic_dataloaders
# from cbo_in_python.src.torch.optimizer import Optimizer
# from cbo_in_python.src.torch.loss import Loss
from torch.utils.data import Dataset, DataLoader

from collections import OrderedDict
from pyDOE2 import fullfact

MODELS = {
    'MNIST_726x10': MNIST_726x10,
    'MNIST_726x20': MNIST_726x20,
    'MNIST_726x10x10': MNIST_726x10x10,
    'PARA_5x5x5' : PARA_5x5x5,
    'PARA_7x7': PARA_7x7,
    'PARA_25': PARA_25,
    'LeNet1': LeNet1,
    'LeNet5': LeNet5,
    'Net': Net
}

DATASETS = {
    'MNIST': load_mnist_dataloaders,
    'PARABOLA': load_parabola_dataloaders
}

def _evaluate_class(model, data, target, loss_fn):
    # with torch.no_grad():
    outputs = model(data)
    y_pred = torch.argmax(outputs, dim=1)
    loss = loss_fn(outputs, target)
    acc = 1. * target.eq(y_pred).sum().item() / target.shape[0]
    return loss, acc

def _evaluate_reg(model, data, target, loss_fn):
    # with torch.no_grad():
    outputs = model(data)
    loss = loss_fn(outputs, target)
    return loss

def number_of_nn_evaluations(n_train_batches,
                             epochs):

    return (n_train_batches)*epochs, n_train_batches*epochs


def train(model, train_dataloader, test_dataloader, device, use_multiprocessing, processes,
          epochs, lr, gamma, log_interval, problem_type):
    train_accuracies = []
    train_losses = []
    test_accuracies = []
    test_losses = []
    forward_times = []
    backward_times = []

    # optimizer = Optimizer(model, n_particles=particles, alpha=alpha, sigma=sigma,
    #                       l=l, dt=dt, anisotropic=anisotropic, eps=eps, partial_update=partial_update,
    #                       use_multiprocessing=use_multiprocessing, n_processes=processes,
    #                       particles_batch_size=particles_batch_size, device=device)

    optimizer = optim.Adadelta(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

    if problem_type == 'classification':
        loss_fn = F.nll_loss
    else:
        loss_fn = F.mse_loss

    n_batches = len(train_dataloader)

    for epoch in range(epochs):
        epoch_train_accuracies = []
        epoch_train_losses = []
        epoch_forward_times = []
        epoch_backward_times = []
        model.train()
        for batch_idx, (data, target) in enumerate(train_dataloader):
            data, target = data.to(device), target.to(device)

            if problem_type == 'classification':
                # with torch.no_grad():
                optimizer.zero_grad()
                start_forward = time.time()
                outputs = model(data)
                y_pred = torch.argmax(outputs, dim=1)
                forward_time = time.time() - start_forward
                loss_train = loss_fn(outputs, target)
                acc_train = 1. * target.eq(y_pred).sum().item() / target.shape[0]
                # return loss, acc
            else:
                optimizer.zero_grad()
                start_forward = time.time()
                loss_train = _evaluate_reg(model, data, target, loss_fn)
                forward_time = time.time() - start_forward
                acc_train = 0.0

            start_backward = time.time()
            loss_train.backward()
            backward_time = time.time() - start_backward
            optimizer.step()

            epoch_train_accuracies.append(acc_train)
            epoch_train_losses.append(loss_train.cpu().item())
            epoch_forward_times.append(forward_time)
            epoch_backward_times.append(backward_time)

            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_dataloader.dataset),
                    100. * batch_idx / len(train_dataloader), loss_train.item()))

        train_accuracies.append(np.mean(epoch_train_accuracies))
        train_losses.append(np.mean(epoch_train_losses))
        forward_times.append(np.sum(epoch_forward_times))
        backward_times.append(np.sum(epoch_backward_times))

        with open(run_file, 'a') as file:
            file.write('\nTrain Epoch: {} \tLoss: {:.6f}, Forward Time: {:.3f}, Backward Time: {:.3f}'.format(
                epoch, train_losses[-1], forward_times[-1], backward_times[-1]))

        model.eval()
        with torch.no_grad():
            losses = []
            accuracies = []
            for X_test, y_test in test_dataloader:
                X_test, y_test = X_test.to(device), y_test.to(device)
                if problem_type == 'classification':
                    loss_test, acc_test = _evaluate_class(model, X_test, y_test, loss_fn)
                else:
                    loss_test = _evaluate_reg(model, X_test, y_test, loss_fn)
                    acc_test = 0.0
                losses.append(loss_test.cpu().item())
                accuracies.append(acc_test)
            loss_test, acc_test = np.mean(losses), np.mean(accuracies)
            test_accuracies.append(acc_test)
            test_losses.append(loss_test)
        lr = optimizer.param_groups[0]['lr']
        print('\nTest set: Average loss: {:.4f}, Accuracy: ({:.0f}%)\n'.format(loss_test, acc_test*100))
        with open(run_file, 'a') as file:
            file.write('\nTest set: Average Loss: {:.4f}, Accuracy: ({:.0f}%)\n'.format(loss_test, acc_test*100))

        # print(
        #     f'Epoch: {epoch + 1:2}/{epochs}, batch: {batch_idx + 1:4}/{n_batches}, train loss: {loss_train:8.3f}, '
        #     f'train acc: {acc_train:8.3f}, test loss: {val_loss:8.3f}, test acc: {val_acc:8.3f}, lr: {lr:8.3f}, gamma: {scheduler.gamma:8.3f}',
        #     flush=True)

        scheduler.step()

    # if save_model:
    #     torch.save(model.state_dict(), "mnist_cnn.pt")
        # if cooling:
        #     optimizer.cooling_step()

    return train_accuracies, test_accuracies, train_losses, test_losses, forward_times, backward_times


def build_plot(epochs, model_name, dataset_name, plot_path,
               train_acc, test_acc, train_loss, test_loss):
    plt.rcParams['figure.figsize'] = (20, 10)
    plt.rcParams['font.size'] = 25

    epochs_range = np.arange(1, epochs + 1, dtype=int)

    plt.clf()
    fig, (ax1, ax2) = plt.subplots(1, 2)

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

    plt.suptitle(f'{model_name} @ {dataset_name}')
    plt.savefig(plot_path)

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

def create_results_directory(directory, results_directory_name=None):
    if results_directory_name is None:
        now = datetime.datetime.now()
        doe_date = '-'.join([str(now.year), str(now.month), str(now.day)]) + '_' + '-'.join([str(now.hour), str(now.minute)])
        results_directory = os.path.join(directory, doe_date)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--model', type=str, default='Net', help=f'architecture to use',
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

    parser.add_argument('--epochs', type=int, default=3, help='train for EPOCHS epochs')
    parser.add_argument('--batch_size', type=int, default=60, help='batch size (for samples-level batching)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')

    parser.add_argument('--build_plot', required=False, action='store_true',
                        help='specify to build loss and accuracy plot')
    parser.add_argument('--plot_path', required=False, type=str, default='grad_demo.png',
                        help='path to save the resulting plot')

    parser.add_argument('--log_interval', type=int, default=10, help='evaluate test accuracy every LOG_INTERVAL '
                                                                     'samples-level batches')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--results_directory_name', required=False, type=str, default='GRAD_MNIST_RESULTS',
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

    # Code inspired by https://github.com/pytorch/examples/blob/main/mnist/main.py#L114

    train_dataloader, test_dataloader = DATASETS[args.dataset](train_batch_size=args.batch_size,
                                                               test_batch_size=1000)

    if args.dataset == 'PARABOLA':
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

        if args.dataset == 'PARABOLA':
            doe_models = ['PARA_5x5x5', 'PARA_7x7', 'PARA_25']
            doe_epochs = [5, 10, 15]
        else:
            doe_models = ['MNIST_726x10', 'MNIST_726x10x10', 'MNIST_726x20']
            doe_epochs = [5, 10, 15]


        doe_parameters = OrderedDict({'models': doe_models,
                                      'epochs': doe_epochs})

        experiments = create_doe_experiments(doe_parameters, method='fullfact')

        experiment_result_by_epochs = {}
        for doe_epoch in doe_epochs:
            experiment_result_by_epochs['{}'.format(doe_epoch)] = {}

        for averaging_run in range(1, args.n_runs+1):

            for n_exp, experiment in enumerate(experiments):
                nn_model = experiment['models']
                epochs = experiment['epochs']
                model = MODELS[nn_model]()
                print(f'Training {nn_model} @ {args.dataset} (Run {averaging_run})')
                forward_evals, backward_evals = number_of_nn_evaluations(len(train_dataloader),
                                                    epochs)
                # print(f'Number of forward NN evaluations: {forward_evals}')
                # print(f'Number of backward NN evaluations: {backward_evals}')
                trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

                experiment_directory = create_experiment_directory(results_directory, n_exp)
                run_file = os.path.join(experiment_directory, 'results.txt')
                parameter_file_name = '{}_{}.pt'.format(args.dataset, nn_model)
                parameter_file = os.path.join(experiment_directory, parameter_file_name)
                with open(run_file, 'a') as file:
                    file.write('\n' + str(experiment))
                    file.write('\nTrainable Parameters: {}'.format(trainable_parameters))

                start_time = time.time()
                result = train(model, train_dataloader, test_dataloader, device, use_multiprocessing, args.processes,
                            epochs, args.lr, args.gamma, args.log_interval, problem_type=problem_type)
                elapsed_time = time.time() - start_time

                accuracies_train, accuracies_test, losses_train, losses_test, forward_times, backward_times = result
                forward_time = np.sum(forward_times)
                backward_time = np.sum(backward_times)
                forward_time_one = forward_time/forward_evals
                effective_forward_evals = forward_evals + backward_time/forward_time_one

                print('Elapsed time: {:.1f} seconds, Forward: {:.3f}, Backward: {:.3f}'.format(
                    elapsed_time, forward_time, backward_time))

                best_epoch = np.argmin(np.array(losses_test))
                best_accuracy_test = accuracies_test[best_epoch]
                best_loss_test = losses_test[best_epoch]

                if nn_model not in experiment_result_by_epochs['{}'.format(epochs)].keys():
                    experiment_result_by_epochs['{}'.format(epochs)][nn_model]={'loss': [],
                                                                                'acc': [],
                                                                                'forward_evals': [],
                                                                                'time': []}

                experiment_result_by_epochs['{}'.format(epochs)][nn_model]['loss'].append(losses_test)
                experiment_result_by_epochs['{}'.format(epochs)][nn_model]['acc'].append(accuracies_test)
                experiment_result_by_epochs['{}'.format(epochs)][nn_model]['forward_evals'].append(effective_forward_evals)
                experiment_result_by_epochs['{}'.format(epochs)][nn_model]['time'].append(elapsed_time)

                # if args.build_plot:
                #     result = result[0:4]
                #     build_plot(epochs, args.model, args.dataset, os.path.join(results_directory, 'loss_' + args.plot_path),*result)

                with open(results_file, 'a') as file:
                    file.write('''\nExperiment {}/{}, Best Epoch: {}/{}, Best Loss: {:.4f}, Best Accuracy: ({:.0f}%), Final Loss: {:.4f}, Final Accuracy: ({:.0f}%), NN-Evaluations (forw./backw.): {}/{} ({:.3f}s/{:.3f}s), Par. {} Time: {:.1f}'''.format(
                        n_exp, len(experiments)-1, best_epoch, epochs, best_loss_test,
                        best_accuracy_test*100, losses_test[-1], accuracies_test[-1]*100,
                        forward_evals, backward_evals, forward_time, backward_time,
                        trainable_parameters, elapsed_time
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
                        # experiment_model = experiment['NN']
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

                # PLOTTING THE RESULTS FOR ALL EXPERIMENTS; WEIGHT THE PERFORMANCE BY NUMBER OF #PARAMETERS
                for epoch, experiment_results in experiment_result_by_epochs.items():
                    title = 'Results for {} Epochs ({} runs)'.format(epoch, averaging_run)
                    plt.rcParams['figure.figsize'] = (20, 10)
                    plt.rcParams['font.size'] = 25
                    epochs_range = np.arange(1, int(epoch) + 1, dtype=int)
                    plt.clf()
                    fig, (ax1, ax2) = plt.subplots(1, 2)

                    for experiment_model, experiment in experiment_results.items():
                        # experiment_model = experiment['NN']
                        loss_history = np.asarray(experiment['loss']).mean(axis=0)
                        acc_history = np.asarray(experiment['acc']).mean(axis=0)
                        label = '{}'.format(experiment_model)

                        ax1.plot(epochs_range, np.asarray(acc_history)/trainable_parameters, label=label)
                        ax1.legend()
                        ax1.set_xlabel('Epoch')
                        ax1.set_ylabel('Accuracy')
                        ax1.set_title('Accuracy per #Parameters')

                        ax2.plot(epochs_range, np.asarray(loss_history)*trainable_parameters, label=label)
                        ax2.legend()
                        ax2.set_xlabel('Epoch')
                        ax2.set_ylabel('Loss')
                        ax2.set_title('#Parameters weighted Loss')
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
                evals_ax.set_title('Effective Forward Evaluations ({} runs)'.format(averaging_run))
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
        forward_evals, backward_evals = number_of_nn_evaluations(len(train_dataloader),
                                            args.epochs)
        print(f'Number of forward NN evaluations: {forward_evals}')
        print(f'Number of backward NN evaluations: {backward_evals}')
        start_time = time.time()
        result = train(model, train_dataloader, test_dataloader, device, use_multiprocessing, args.processes,
                    args.epochs, args.lr, args.gamma, args.log_interval, problem_type=problem_type, save_model=args.save_model)
        print(f'Elapsed time: {time.time() - start_time} seconds')
        if args.build_plot:
            build_plot(args.epochs, args.model, args.dataset, os.path.join(results_directory, 'loss_' + args.plot_path),
                    *result)

            # result_plot(args.model, args.dataset, 'predictions_' + args.plot_path, train_dataloader.dataset.x,
            #             model(train_dataloader.dataset.x).detach().numpy(), test_dataloader.dataset.x,
            #             model(test_dataloader.dataset.x).detach().numpy(), train_dataloader.dataset.x+test_dataloader.dataset.x,
            #             train_dataloader.dataset.y+test_dataloader.dataset.y)





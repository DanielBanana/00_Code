import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import argparse
import time
import matplotlib.pyplot as plt


sys.path.insert(1, os.getcwd())
from fmu_helper import FMUEvaluator
from cbo_in_python.src.torch_.models import *
# from cbo_in_python.src.datasets import load_mnist_dataloaders, load_parabola_dataloaders, f
from cbo_in_python.src.datasets import create_generic_dataset, load_generic_dataloaders
from cbo_in_python.src.torch_.optimizer import Optimizer
from cbo_in_python.src.torch_.loss import Loss
from torch.utils.data import Dataset, DataLoader

from utils import build_plot, result_plot

MODELS = {
    'SimpleMLP': SimpleMLP,
    'TinyMLP': TinyMLP,
    'SmallMLP': SmallMLP,
    'LeNet1': LeNet1,
    'LeNet5': LeNet5,
}

DATASETS = {
    'VdP': ''
}

# # The Neural Network structure class
class VdPMLP(nn.Module):
    def __init__(self):
        super(VdPMLP, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(2, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),

            nn.Linear(20, 1)
        )
        self.double()

    def forward(self, x):
        return self.model(x)


class Hybrid(nn.Module):
    def __init__(self, fmu_model, augment_model, z0, t):
        super(Hybrid, self).__init__()

        self.fmu_model = fmu_model
        self.augment_model = augment_model
        self.z0 = z0
        self.t = t

    def augment_model_function(self, augment_model_parameters, input):
        # The augment_model is currently a pytorch model, which just takes
        # the input. It has its own parameters saved internally.
        # The f_euler function expects a model which needs its paramaters
        # given when it is called: y = augment_model_function(parameters, input)
        # f_euler provides the input to the augment_model as numpy array
        # but we can only except tensors, so convert
        return self.augment_model(torch.tensor(input)).detach().numpy()

    def forward(self, pointers):
        '''Applies euler to the VdP ODE by calling the fmu; returns the trajectory'''
        t = self.t
        z0 = self.z0
        z = np.zeros((t.shape[0], 2))
        z[0] = z0
        # Forward the initial state to the FMU
        self.fmu_model.setup_initial_state(z0, pointers)
        times = []
        for i in range(len(t)-1):
            # start = time.time()
            status = self.fmu_model.fmu.setTime(t[i])
            dt = t[i+1] - t[i]

            pointers, enterEventMode, terminateSimulation = self.fmu_model.evaluate_fmu(t[i], dt, self.augment_model_function, None, pointers)

            z[i+1] = z[i] + dt * pointers.dx

            if terminateSimulation:
                break

        return z

# For calculation of the reference solution we need the correct behaviour of the VdP
def damping(mu, inputs):
    return mu * (1 - inputs[0]**2) * inputs[1]

def _evaluate_reg(outputs, y_, loss_fn):
    with torch.no_grad():
        loss = loss_fn(outputs, y_)
    return loss

def train(hybrid_model:Hybrid, train_dataloader, test_dataloader, device, use_multiprocessing, processes,
          epochs, particles, particles_batch_size,
          alpha, sigma, l, dt, anisotropic, eps, partial_update, cooling,
          eval_freq, problem_type, pointers):
    train_accuracies = []
    train_losses = []
    test_accuracies = []
    test_losses = []

    # Optimizes the Neural Network with CBO
    augment_optimizer = Optimizer(hybrid_model, n_particles=particles, alpha=alpha, sigma=sigma,
                          l=l, dt=dt, anisotropic=anisotropic, eps=eps, partial_update=partial_update,
                          use_multiprocessing=use_multiprocessing, n_processes=processes,
                          particles_batch_size=particles_batch_size, device=device)



    augment_model_parameters = []

    if problem_type == 'classification':
        pass
        # loss_fn = Loss(F.nll_loss, optimizer)
    else:
        # loss_fn = Loss(F.mse_loss, augment_optimizer)
        loss_fn = Loss(F.mse_loss, augment_optimizer)

    n_batches = len(train_dataloader)

    for epoch in range(epochs):
        epoch_train_accuracies = []
        epoch_train_losses = []
        for batch, (X, y) in enumerate(train_dataloader):
            X_train, y_train = X.to(device), y.to(device)

            # Calculate current solution
            hybrid_model.t = X_train.detach().numpy()
            hybrid_model.z0 = y_train[0]
            z = torch.tensor(hybrid_model(pointers))

            if problem_type == 'classification':
                pass
                # train_loss, train_acc = _evaluate_class(model, X, y, F.nll_loss)
            else:
                train_loss = _evaluate_reg(z, y_train, F.mse_loss)
                train_acc = 0.0
            epoch_train_accuracies.append(train_acc)
            epoch_train_losses.append(train_loss.cpu())

            augment_optimizer.zero_grad()
            loss_fn.backward(z, y, backward_gradients=False)
            augment_optimizer.step()

            if batch % eval_freq == 0 or batch == n_batches - 1:
                with torch.no_grad():
                    losses = []
                    accuracies = []
                    for X_test, y_test in test_dataloader:
                        X_test, y_test = X_test.to(device), y_test.to(device)
                        if problem_type == 'classification':
                            pass
                            # loss, acc = _evaluate_class(model, X_test, y_test, F.nll_loss)
                        else:
                            hybrid_model.z0 = y_test[0]
                            hybrid_model.t = X_test.detach().numpy()
                            z = torch.tensor(hybrid_model(pointers))
                            loss = _evaluate_reg(z, y_test, F.mse_loss)
                            acc = 0.0
                            losses.append(loss.cpu())
                            accuracies.append(acc)
                    val_loss, val_acc = np.mean(losses), np.mean(accuracies)
                    if batch == n_batches - 1:
                        test_accuracies.append(val_acc)
                        test_losses.append(val_loss)

            print(
                f'Epoch: {epoch + 1:2}/{epochs}, batch: {batch + 1:4}/{n_batches}, train loss: {train_loss:8.3f}, '
                f'train acc: {train_acc:8.3f}, test loss: {val_loss:8.3f}, test acc: {val_acc:8.3f}, alpha: {augment_optimizer.alpha:8.3f}, sigma: {augment_optimizer.sigma:8.3f}',
                flush=True)

        train_accuracies.append(np.mean(epoch_train_accuracies))
        train_losses.append(np.mean(epoch_train_losses))
        if cooling:
            augment_optimizer.cooling_step()

    return train_accuracies, test_accuracies, train_losses, test_losses

def f_euler(z0, t, fmu_evaluator: FMUEvaluator, model, model_parameters, pointers):
    '''Applies euler to the VdP ODE by calling the fmu; returns the trajectory'''
    z = np.zeros((t.shape[0], 2))
    z[0] = z0
    # Forward the initial state to the FMU
    fmu_evaluator.setup_initial_state(z0, pointers)
    times = []
    if fmu_evaluator.training:
        dfmu_dz_trajectory = []
        dfmu_dinput_trajectory = []
    for i in range(len(t)-1):
        # start = time.time()
        status = fmu_evaluator.fmu.setTime(t[i])
        dt = t[i+1] - t[i]

        pointers, enterEventMode, terminateSimulation = fmu_evaluator.evaluate_fmu(t[i], dt, model, model_parameters, pointers)

        z[i+1] = z[i] + dt * pointers.dx

        if terminateSimulation:
            break

    return z

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', type=str, default='SimpleMLP', help=f'architecture to use',
                        choices=list(MODELS.keys()))
    parser.add_argument('--dataset', type=str, default='VdP', help='dataset to use',
                        choices=list(DATASETS.keys()))
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='cuda',
                        help='whether to use GPU (cuda) for accelerated computations or not')
    parser.add_argument('--use_multiprocessing', action='store_true',
                        help='specify to use multiprocessing for accelerating computations on CPU '
                             '(note, it is impossible to use multiprocessing with GPU)')
    parser.add_argument('--processes', type=int, default=4,
                        help='how many processes to use for multiprocessing')

    parser.add_argument('--epochs', type=int, default=100, help='train for EPOCHS epochs')
    parser.add_argument('--batch_size', type=int, default=20, help='batch size (for samples-level batching)')
    parser.add_argument('--particles', type=int, default=1000, help='')
    parser.add_argument('--particles_batch_size', type=int, default=50, help='batch size '
                                                                             '(for particles-level batching)')

    parser.add_argument('--alpha', type=float, default=100, help='alpha from CBO dynamics')
    parser.add_argument('--sigma', type=float, default=0.1 ** 0.5, help='sigma from CBO dynamics')
    parser.add_argument('--l', type=float, default=1, help='lambda from CBO dynamics')
    parser.add_argument('--dt', type=float, default=0.1, help='dt from CBO dynamics')
    parser.add_argument('--anisotropic', type=bool, default=True, help='whether to use anisotropic or not')
    parser.add_argument('--eps', type=float, default=1e-4, help='threshold for additional random shift')
    parser.add_argument('--partial_update', type=bool, default=True, help='whether to use partial or full update')
    parser.add_argument('--cooling', type=bool, default=True, help='whether to apply cooling strategy')

    parser.add_argument('--build_plot', required=False, default=True, action='store_true',
                        help='specify to build loss and accuracy plot')
    parser.add_argument('--plot_path', required=False, type=str, default='demo.png',
                        help='path to save the resulting plot')

    parser.add_argument('--eval_freq', type=int, default=100, help='evaluate test accuracy every EVAL_FREQ '
                                                                   'samples-level batches')
    args = parser.parse_args()
    # ODE SETUP
    ####################################################################################
    # Training Setup
    train_Tstart = 0.0
    train_Tend = 100.0
    train_nSteps = 5001
    train_t = np.linspace(train_Tstart, train_Tend, train_nSteps)
    train_z0 = np.array([1.0, 0.0])

    # Test Setup
    test_Tstart = train_Tend
    test_Tend = train_Tend + (train_Tend - train_Tstart)*0.5
    test_nSteps = int(train_nSteps * 0.5)
    test_t = np.linspace(test_Tstart, test_Tend, test_nSteps)

    mu = 5.0

    # FMU SETUP
    ####################################################################################
    fmu_filename = 'Van_der_Pol_damping_input.fmu'
    path = os.path.abspath(__file__)
    fmu_filename = '/'.join(path.split('/')[:-1]) + '/' + fmu_filename
    fmu_evaluator = FMUEvaluator(fmu_filename, train_Tstart, train_Tend)
    pointers = fmu_evaluator.get_pointers()

    train_z = f_euler(z0=train_z0, t=train_t, fmu_evaluator=fmu_evaluator, model=damping, model_parameters=mu, pointers=pointers)
    fmu_evaluator.reset_fmu(test_Tstart, test_Tend)
    test_z = f_euler(z0=train_z[-1], t=test_t, fmu_evaluator=fmu_evaluator, model=damping, model_parameters=mu, pointers=pointers)
    fmu_evaluator.reset_fmu(train_Tstart, train_Tend)

    # CONVERT THE REFERENCE DATA TO A DATASET
    ####################################################################################

    train_dataset = create_generic_dataset(torch.tensor(train_t), torch.tensor(train_z))
    test_dataset = create_generic_dataset(torch.tensor(test_t), torch.tensor(test_z))

    train_dataloader, test_dataloader = load_generic_dataloaders(train_dataset=train_dataset,
                                                                 train_batch_size=train_nSteps,
                                                                 test_dataset=test_dataset,
                                                                 test_batch_size=test_nSteps)

    # TORCH DEVICES
    ####################################################################################
    device = args.device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print('Cuda is unavailable. Using CPU instead.')
        device = 'cpu'
    use_multiprocessing = args.use_multiprocessing
    if device != 'cpu' and use_multiprocessing:
        print('Unable to use multiprocessing on GPU')
        use_multiprocessing = False
    device = torch.device(device)


    # TRAINING
    ####################################################################################
    augment_model = VdPMLP()
    hybrid_model = Hybrid(fmu_evaluator, augment_model, train_z0, train_t)
    start_time = time.time()
    result = train(hybrid_model, train_dataloader, test_dataloader, device, use_multiprocessing, args.processes,
                   args.epochs, args.particles, args.particles_batch_size,
                   args.alpha, args.sigma, args.l, args.dt, args.anisotropic, args.eps, args.partial_update,
                   args.cooling,
                   args.eval_freq,
                   problem_type='regression',
                   pointers=pointers)
    print(f'Elapsed time: {time.time() - start_time} seconds')
    if args.build_plot:
        build_plot(args.epochs, args.model, args.dataset, 'loss_' + args.plot_path,
                   *result)

        X_train = train_dataloader.dataset.x
        X_test = test_dataloader.dataset.x
        hybrid_model.t = X_train.detach().numpy()
        hybrid_model.z0 = train_dataloader.dataset.y[0]
        z_train = torch.tensor(hybrid_model(pointers))

        hybrid_model.z0 = test_dataloader.dataset.y[0]
        hybrid_model.t = X_test.detach().numpy()
        z_test = torch.tensor(hybrid_model(pointers))
        result_plot(args.model, args.dataset, 'predictions_' + args.plot_path, X_train,
                    z_train[:,0], X_test,
                    z_test[:,0], np.hstack((train_t, test_t)), np.vstack((train_z, test_z))[:,0])
        result_plot(args.model, args.dataset, 'predictions_' + args.plot_path, X_train,
                    z_train[:,1], X_test,
                    z_test[:,1], np.hstack((train_t, test_t)), np.vstack((train_z, test_z))[:,1])
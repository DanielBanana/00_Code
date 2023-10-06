import argparse

from functools import partial
import jax.numpy as jnp
from jax import lax
import logging
import numpy as np
import orbax
import os
import torch
from scipy.optimize import minimize
import sys
import time
import yaml

sys.path.insert(1, os.getcwd())
from cbo_in_python.src.datasets import create_generic_dataset, load_generic_dataloaders
from cbo_in_python.src.torch_.models import *
from VdP import VdP
from utils import create_results_directory, create_results_subdirectories, create_residual_reference_solution
import ASM
import CBO


namespaces = {'VdP': VdP}


def create_reference_solution(start, end, n_steps, z0, reference_ode_parameters, ode_integrator):
    t_train = np.linspace(start, end, n_steps)
    z_ref_train = np.asarray(ode_integrator(z0, t_train, reference_ode_parameters))
    # Generate the reference data for testing
    z0_test = z_ref_train[-1]
    n_steps_test = int(n_steps * 0.5)
    t_test = np.linspace(end, (end - start) * 1.5, n_steps_test)
    z_ref_test = np.asarray(ode_integrator(z0_test, t_test, reference_ode_parameters))
    return t_train, z_ref_train, t_test, z_ref_test, z0_test

def euler(z0, t, ode_parameters):
    '''Applies forward Euler to the original ODE and returns the trajectory'''
    z = jnp.zeros((t.shape[0], z0.shape[0]))
    z = z.at[0].set(z0)
    i = jnp.asarray(range(t.shape[0]))
    euler_body_func = partial(step, t=t, ode_parameters = ode_parameters)
    final, result = lax.scan(euler_body_func, z0, i)
    z = z.at[1:].set(result[:-1])
    return z

def step(prev_z, i, t, ode_parameters):
    t = jnp.asarray(t)
    dt = t[i+1] - t[i]
    next_z = prev_z + dt * ode(prev_z, t[i], ode_parameters)
    return next_z, next_z

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # PROBLEM SETUP
    parser.add_argument('--problemDescription', type=str, default='Problem_VdP.yaml', help='''Yaml file of the
                        the problem setup''')
    parser.add_argument('--datamodelDescription', type=str, default='Model_NN.yaml', help='''Yaml file of the
                        the datadriven model setup''')
    parser.add_argument('--optimizerDescription', type=str, default='Optimizer_CBO.yaml', help='''Yaml file of the
                        the optimizer setup''')
    parser.add_argument('--gS', type=str, default='GeneralSettings.yaml', help='''Where to save the results
                        and what to plot''')
    args = parser.parse_args()

    # Find out the directory of the current file. It is assumed all neccessary files
    # are also in this directory
    path = os.path.abspath(__file__)
    directory = os.path.sep.join(path.split(os.path.sep)[:-1])
    pD_path = os.path.join(directory, args.problemDescription)
    mD_path = os.path.join(directory, args.datamodelDescription)
    oD_path = os.path.join(directory, args.optimizerDescription)
    gS_path = os.path.join(directory, args.gS)

    # TODO logger

    # Read out the problem description:
    with open(pD_path, 'r') as file:
        try:
            pD = yaml.safe_load(file)
        except yaml.YAMLError as e:
            print("Error reading YAML:", e)
            exit(2)

    # Read out the model description:
    with open(mD_path, 'r') as file:
        try:
            mD = yaml.safe_load(file)
        except yaml.YAMLError as e:
            print("Error reading YAML:", e)
            exit(2)

    # Read out the optimizer description:
    with open(oD_path, 'r') as file:
        try:
            oD = yaml.safe_load(file)
        except yaml.YAMLError as e:
            print("Error reading YAML:", e)
            exit(2)

    # Read out the problem description:
    with open(gS_path, 'r') as file:
        try:
            gS = yaml.safe_load(file)
        except yaml.YAMLError as e:
            print("Error reading YAML:", e)
            exit(2)


    results_directory = create_results_directory(directory=directory, results_directory_name=gS['results_directory'])
    logger_name = '_'.join([pD['name'], mD['name'], oD['name']])
    log_file = os.path.join(results_directory, logger_name+'.log')
    logging.basicConfig(filename=log_file, encoding='utf-8', level=logging.INFO)
    logger = logging.getLogger(logger_name)

    epoch = 0
    x0_train = np.array(pD['ic'])

    if oD['name'] == 'CBO':
        gS['device'], gS['use_multiprocessing'] = CBO.determine_device(gS['device'], gS['use_multiprocessing'])


    if pD['fmu']:
        pass
    else:
        # 1. Create Reference Solution
        # A reference solution is needed for all problems, models and optimizers
        logger.info('Creating Reference Solution')
        namespace = namespaces[pD['file']]
        ode = namespace.ode
        ode_res = namespace.ode_res
        ode_hybrid = namespace.ode_hybrid
        t_train, x_train_ref, t_test, x_test_ref, x0_test = create_reference_solution(
            pD['integration_parameters']['start'],
            pD['integration_parameters']['end'],
            pD['integration_parameters']['steps'],
            x0_train,
            pD['variables'],
            ode_integrator=euler)

        reference_data = {
            't_train': t_train,
            'x_train_ref': x_train_ref,
            't_test': t_test,
            'x_test_ref': x_test_ref
        }

        if oD['residual_training']:
            residual_directory, checkpoint_directory = create_results_subdirectories(results_directory=results_directory, trajectory=False, residual=True)

            x_train_res, y_train_res, x_test_res, y_test_res = create_residual_reference_solution(t_train, x_train_ref, t_test, x_test_ref, pD['variables'], ode_res)

            if mD['name'] == 'NN':
                if oD['name'] == 'CBO':
                    layers = [pD['inputs']] + mD['layers'] + [pD['outputs']]
                    datamodel = CustomMLP(layers)
                    hybrid_model = CBO.Hybrid_Python(pD['variables'], datamodel, x0_train, t_train, ode_hybrid, mode='residual')
                elif oD['name'] == 'ASM':
                    layers = mD['layers'] + [pD['outputs']]
                    datamodel, parameters = ASM.create_nn(layers, x0_train)
                    hybrid_model = None
                else:
                    logger.error(f'Unknown Model: {oD["name"]}')
                    exit(1)

                checkpoint_manager_save = None

                if gS['save_parameters']:
                    checkpointer_save = orbax.checkpoint.PyTreeCheckpointer()
                    options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=5, create=True)
                    checkpoint_manager_save = orbax.checkpoint.CheckpointManager(checkpoint_directory, checkpointer_save, options)
                else:
                    checkpoint_manager_save = None

                if gS['load_parameters']:
                    load_directory = os.path.join(directory, gS['load_name'])
                    checkpointer_load = orbax.checkpoint.PyTreeCheckpointer()
                    options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=5, create=True)
                    checkpoint_manager_load = orbax.checkpoint.CheckpointManager(restore_directory, checkpointer_load, options)
                    step = checkpoint_manager_load.latest_step()
                    parameters = checkpoint_manager_load.restore(step)
                    if oD['name'] == 'CBO':
                        datamodel = CBO.load_jax_parameters(datamodel, parameters)

                # Arguments for ASM and CBO
                optimizer_args = {'problemDescription': pD,
                    'modelDescription': mD,
                    'optimizerDescription': oD,
                    'generalSettings': gS,
                    'epochs': oD['epochs_res'],
                    'reference_data': reference_data,
                    'losses_train' : [],
                    'losses_train_res': [],
                    'losses_test': [],
                    'losses_test_res': [],
                    'accuracies_train': [],
                    'accuracies_test': [],
                    'best_parameters': None,
                    'best_loss': np.inf,
                    'best_loss_test': np.inf,
                    'best_pred': None,
                    'best_pred': None,
                    'checkpoint_manager': checkpoint_manager_save,
                    'logger': logger,
                    'results_directory': residual_directory,
                    'residual': True,
                    'pointers': None # only needed for FMU
                }

                # if pD['type'] == 'classification':
                #     optimizer_args['accuracies_train'] = []
                #     optimizer_args['accuracies_test'] = []

                results_dict = {}

                if oD['name'] == 'CBO':
                    train_dataset = create_generic_dataset(torch.tensor(x_train_res), torch.tensor(y_train_res))
                    test_dataset = create_generic_dataset(torch.tensor(x_test_res), torch.tensor(y_test_res))
                    train_dataloader, test_dataloader = load_generic_dataloaders(train_dataset=train_dataset,
                                                                                train_batch_size=oD['batch_size'] if oD['batch_size'] < len(train_dataset) else len(train_dataset),
                                                                                test_dataset=test_dataset,
                                                                                test_batch_size=oD['batch_size'] if oD['batch_size'] < len(test_dataset) else len(test_dataset),
                                                                                shuffle=False)
                    optimizer_args['train_dataloader'] = train_dataloader
                    optimizer_args['test_dataloader'] = test_dataloader

                    time_start = time.time()

                    CBO.train(model=hybrid_model, args=optimizer_args)

                    experiment_time = time.time()-time_start
                    datamodel, ckpt = CBO.get_best_parameters(hybrid_model, residual_directory)

                    if gS['plot_prediction']:
                        datamodel.t = t_train
                        datamodel.z0 =  x0_train
                        datamodel.trajectory_mode()
                        x_train_pred = datamodel()
                        datamodel.t = t_test
                        datamodel.z0 = x0_test
                        x_test_pred = datamodel()

                elif oD['name'] == 'ASM':
                    x0s, targets, ts = ASM.create_clean_mini_batch(oD['clean_batches'], x_train_ref, t_train)
                    optimizer_args['z0s'] = x0s
                    optimizer_args['targets'] = targets
                    optimizer_args['ts'] = ts
                    flat_parameters, unravel_pytree = flatten_util.ravel_pytree(parameters)
                    optimizer_args['unravel_function'] = unravel_pytree

                    time_start = time.time()

                    if oD['method'] == 'Adam':
                        Adam = ASM.AdamOptim(eta=args.adam_eta, beta1=args.adam_beta1, beta2=args.adam_beta2, epsilon=args.adam_eps)
                        for i in range(args.res_steps):
                            loss, grad = residual_wrapper(flat_nn_parameters=flat_parameters, optimizer_args=optimizer_args)
                            flat_nn_parameters = Adam.update(i+1, flat_parameters, grad)
                    elif oD['method'] in ['CG', 'BFGS', 'SLSQP']:
                        minimize(ASM.residual_wrapper, flat_parameters, method=oD['method'], jac=True, args=optimizer_args, options={'maxiter':oD['epochs'], 'tol': oD['tol']})

                    run_time = time.time() - time_start

                    best_parameters = optimizer_args['best_parameters']
                    best_loss = optimizer_args['best_loss']
                    best_loss_test = optimizer_args['best_loss_test']
                    x_train_best = optimizer_args['best_pred']
                    x_test_best = optimizer_args['best_pred_test']

                    flat_parameters, _ = flatten_util.ravel_pytree(best_parameters)

                    logger.info(f'Best Loss in Residual Training: {best_loss}')

                else:
                    logger.error(f'Unknown Model: {oD["name"]}')
                    exit(1)


                if gS['plot_prediction']:
                    result_plot_multi_dim(mD['name'], pd['name'], os.path.join(residual_directory, f'Final.png'),
                    t_train, x_train_pred, t_test, x_test_pred, np.hstack((t_train, t_test)), np.vstack((x_train_ref, x_test_ref)))

                if gS['plot_loss']:
                    build_plot(epochs=optimizer_args['epochs'],
                        model_name=mD['name'],
                        dataset_name=pd['name'],
                        plot_path=os.path.join(results_directory, 'Loss.png'),
                        train_acc=optimizer_args['accuracies_train'],
                        test_acc=optimizer_args['accuracies_test'],
                        train_loss=optimizer_args['losses_train'],
                        test_loss=optimizer_args['losses_test']
                    )

                if gS['plot_parameters']:
                    pass


                    # plot_results(t_train, x_best, x_train_ref, residual_path+'_best')
                    # plot_losses(epochs=list(range(len(optimizer_args['losses']))), training_losses=optimizer_args['losses'], validation_losses=optimizer_args['losses_val'], path=residual_path+'_losses')
                    # plot_losses(epochs=list(range(len(optimizer_args['losses_res']))), training_losses=optimizer_args['losses_res'], path=residual_path+'_losses_res')

                    # results_dict['losses_train_res'] =  list(optimizer_args['losses_res'])
                    # results_dict['time_res'] = experiment_time



                else:
                    logger.error(f'Unknown Model: {oD["name"]}')
                    exit(1)






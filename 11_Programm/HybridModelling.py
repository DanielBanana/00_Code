import argparse

from functools import partial
import jax
import jax.numpy as jnp
from jax import lax, flatten_util, jit
import logging
import numpy as np
import orbax
import os
import torch
from scipy.optimize import minimize
import sys
import time
import yaml
import shutil

sys.path.insert(1, os.getcwd())
from cbo_in_python.src.datasets import create_generic_dataset, load_generic_dataloaders
from cbo_in_python.src.torch_.models import *
from fmu_helper import FMUEvaluator
from create_NN_FMU import create_NN_FMU
from VdP import VdP
from utils import create_results_directory, create_results_subdirectories, create_residual_reference_solution, result_plot_multi_dim, build_plot, create_residual_reference_dataset_fmu, visualise_wb
import ASM
import CBO
import NN
import CSTR
import Lorenz

namespaces = {'VdP': VdP, 'CSTR': CSTR, 'Lorenz': Lorenz}

def create_reference_solution(start, end, n_steps, x0, variables, ode_integrator, ode, test_factor = 1):
    t_train = np.linspace(start, end, n_steps)
    z_ref_train = np.asarray(ode_integrator(x0, t_train, variables, ode))
    # Generate the reference data for testing
    x0_test = z_ref_train[-1]
    n_steps_test = int(n_steps * test_factor)
    t_test = np.linspace(end, (end - start) * (1+test_factor), n_steps_test)
    z_ref_test = np.asarray(ode_integrator(x0_test, t_test, variables, ode))
    return t_train, z_ref_train, t_test, z_ref_test, x0_test

def euler(x0, t, variables, ode):
    '''Applies forward Euler to the original ODE and returns the trajectory'''
    z = jnp.zeros((t.shape[0], x0.shape[0]))
    z = z.at[0].set(x0)
    i = jnp.asarray(range(t.shape[0]))
    euler_body_func = partial(step, t=t, variables=variables, ode=ode)
    final, result = lax.scan(euler_body_func, x0, i)
    z = z.at[1:].set(result[:-1])
    return z

def step(prev_z, i, t, variables, ode):
    t = jnp.asarray(t)
    dt = t[i+1] - t[i]
    next_z = prev_z + dt * ode(prev_z, t[i], variables)
    return next_z, next_z

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # PROBLEM SETUP
    parser.add_argument('--pD', type=str, default='Problem_VdP.yaml', help='''Yaml file of the
                        the problem setup''')
    parser.add_argument('--mD', type=str, default='Model_NN.yaml', help='''Yaml file of the
                        the datadriven model setup''')
    parser.add_argument('--oD', type=str, default='Optimizer_ASM.yaml', help='''Yaml file of the
                        the optimizer setup''')
    parser.add_argument('--gS', type=str, default='GeneralSettings.yaml', help='''Yaml file of the
                        the general Settings (where to write files and plots, etc.
                        ''')
    args = parser.parse_args()

    # Find out the directory of the current file. It is assumed all neccessary files
    # are also in this directory
    path = os.path.abspath(__file__)
    directory = os.path.sep.join(path.split(os.path.sep)[:-1])
    pD_path = os.path.join(directory, args.pD)
    mD_path = os.path.join(directory, args.mD)
    oD_path = os.path.join(directory, args.oD)
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

    ####################################################################################
    #### SAVE THE INPUT YAML FILES  ####################################################
    ####################################################################################
    results_directory = create_results_directory(directory=directory, results_directory_name=gS['results_directory'])
    shutil.copy(pD_path, results_directory)
    shutil.copy(mD_path, results_directory)
    shutil.copy(oD_path, results_directory)
    shutil.copy(gS_path, results_directory)

    ####################################################################################
    #### PREPARE LOGGER  ###############################################################
    ####################################################################################
    logger_name = '_'.join([pD['name'], mD['name'], oD['name']])
    log_file = os.path.join(results_directory, logger_name+'.log')
    logging.basicConfig(filename=log_file, encoding='utf-8', level=logging.INFO)
    logger = logging.getLogger(logger_name)

    ####################################################################################
    #### GENERAL PREPARATIONS  #########################################################
    ####################################################################################
    epoch = 0
    test_factor = 1.0

    namespace = namespaces[pD['name']]
    if oD['name'] == 'CBO':
        gS['device'], gS['use_multiprocessing'] = CBO.determine_device(gS['device'], gS['use_multiprocessing'])

    ####################################################################################
    #### OPTIMISATION WITH FMU  ########################################################
    ####################################################################################
    if pD['fmu']:
        ################################################################################
        ## GENERAL FMU PREPARATIONS  ###################################################
        ################################################################################
        fmu_filename = pD['file']
        path = os.path.abspath(__file__)
        fmu_filename = '/'.join(path.split('/')[:-1]) + '/' + fmu_filename
        fmu_evaluator = FMUEvaluator(fmu_filename, pD['integration_parameters']['start'], pD['integration_parameters']['end'])
        state_variables = []
        x0_train = []
        for variable in fmu_evaluator.model_description.modelVariables:
            if variable.name in pD['initial_conditions'].keys():
                state_variables.append(variable.name)
                x0_train.append(pD['initial_conditions'][variable.name])
        x0_train = np.array(x0_train)

        t_train = np.linspace(pD['integration_parameters']['start'], pD['integration_parameters']['end'], pD['integration_parameters']['steps'])
        t_test = np.linspace(pD['integration_parameters']['end'], pD['integration_parameters']['end']*(test_factor+1), int(pD['integration_parameters']['steps']*test_factor))
        reference_training = fmu_evaluator.euler(z0=x0_train, t=t_train, model=namespace.missing_terms, model_parameters=pD['variables'], make_zero_input_prediction=oD['residual_training'])
        fmu_evaluator.reset_fmu(pD['integration_parameters']['end'], pD['integration_parameters']['end']*(test_factor+1))
        if oD['residual_training']:
            x_ref_train, x_ref_train_dot = reference_training
        else:
            x_ref_train = reference_training
        x0_test = x_ref_train[-1]
        reference_test = fmu_evaluator.euler(z0=x0_test, t=t_test, model=namespace.missing_terms, model_parameters=pD['variables'], make_zero_input_prediction=oD['residual_training'])
        fmu_evaluator.reset_fmu(pD['integration_parameters']['start'], pD['integration_parameters']['end'])
        if oD['residual_training']:
            x_ref_test, x_ref_test_dot = reference_test
        else:
            x_ref_test = reference_test

        reference_data = {
            't_train': t_train,
            'x_ref_train': x_ref_train,
            't_test': t_test,
            'x_ref_test': x_ref_test
        }
        if oD['residual_training']:
            ############################################################################
            ## RESIDUAL PREPARATIONS ###################################################
            ############################################################################
            if 'NN' in mD['name']:
                if oD['name'] == 'CBO':
                    layers = [pD['inputs']] + mD['layers'] + [pD['outputs']]
                    datamodel = CustomMLP(layers)
                    hybrid_model = CBO.Hybrid_FMU(fmu_evaluator, datamodel, x0_train, t_train, mode='residual' if oD['residual_training'] else 'trajectory')
                    pointers = hybrid_model.pointers
                    hybrid_model.training = False # only needed for gradient based training
                elif oD['name'] == 'ASM':
                    layers = mD['layers'] + [pD['outputs']]
                    datamodel, parameters, _ = NN.create_nn(layers, x0_train)
                    hybrid_model = None
                    pointers = fmu_evaluator.pointers
                    @jit
                    def J_residual(inputs, outputs, parameters):
                        def squared_error(input, output):
                            pred = datamodel(parameters, input)
                            return (output-pred)**2, pred
                        J, pred = jax.vmap(squared_error)(inputs, outputs)
                        return jnp.mean(J, axis=0)[0], pred
                else:
                    hybrid_model = None
                    datamodel = None
                    logger.error(f'Unknown Model: {oD["name"]}')
                    exit(1)
                if gS['load_parameters']:
                    load_directory = os.path.join(directory, gS['load_name'])
                    checkpointer_load = orbax.checkpoint.PyTreeCheckpointer()
                    options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=5, create=True)
                    checkpoint_manager_load = orbax.checkpoint.CheckpointManager(restore_directory, checkpointer_load, options)
                    step = checkpoint_manager_load.latest_step()
                    parameters = checkpoint_manager_load.restore(step)
                    if oD['name'] == 'CBO':
                        datamodel = CBO.load_jax_parameters(datamodel, parameters)
            else:
                logger.error(f'Unknown Model: {oD["name"]}')
                exit(1)

            residual_directory, checkpoint_directory = create_results_subdirectories(results_directory=results_directory, trajectory=False, residual=True)
            if gS['save_parameters']:
                if oD['name'] == 'CBO':
                    best_parameter_file = os.path.join(checkpoint_directory, gS['save_name'] + '.pt')
                    checkpoint_manager_save = None
                elif oD['name'] == 'ASM':
                    checkpointer_save = orbax.checkpoint.PyTreeCheckpointer()
                    options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=5, create=True)
                    checkpoint_manager_save = orbax.checkpoint.CheckpointManager(checkpoint_directory, checkpointer_save, options)
                    best_parameter_file = None
                else:
                    best_parameter_file = None
                    checkpoint_manager_save = None
            else:
                best_parameter_file = None
                checkpoint_manager_save = None

            # Often the model which we want to improve is already correct in some dimensions
            # We only want to consider the unsatisfactory dims
            x_train_res, y_train_res, x_test_res, y_test_res = create_residual_reference_dataset_fmu(t_train, x_ref_train, x_ref_train_dot, t_test, x_ref_test, x_ref_test_dot)
            y_train_res = y_train_res[:,pD['relevant_output_dims_res']]
            y_test_res = y_test_res[:,pD['relevant_output_dims_res']]

            # Arguments for ASM and CBO
            optimizer_args = {'problemDescription': pD,
                'modelDescription': mD,
                'optimizerDescription': oD,
                'generalSettings': gS,
                'epochs': oD['epochs_res'],
                'epoch': 0,
                'reference_data': reference_data,
                'losses_train' : [],
                'losses_train_res': [],
                'losses_test': [],
                'losses_test_res': [],
                'losses_batches': [], # only needed if we use batching
                'accuracies_train': [],
                'accuracies_test': [],
                'best_parameters': None,
                'best_loss': np.inf,
                'best_loss_test': np.inf,
                'best_pred': None,
                'best_pred_test': None,
                'best_parameter_file': best_parameter_file, # For saving CBO parameters
                'checkpoint_manager': checkpoint_manager_save, # For saving ASM parameters
                'logger': logger,
                'results_directory': residual_directory,
                'residual': True,
                'pointers': pointers # only needed for FMU
            }
            ############################################################################
            ## TRAINING  ###############################################################
            ############################################################################
            train_dataset = create_generic_dataset(torch.tensor(x_train_res), torch.tensor(y_train_res))
            test_dataset = create_generic_dataset(torch.tensor(x_test_res), torch.tensor(y_test_res))
            train_dataloader, test_dataloader = load_generic_dataloaders(train_dataset=train_dataset,
                                                        train_batch_size=oD['batch_size'] if oD['batch_size'] < len(train_dataset) else len(train_dataset),
                                                        test_dataset=test_dataset,
                                                        test_batch_size=oD['batch_size'] if oD['batch_size'] < len(test_dataset) else len(test_dataset),
                                                        shuffle=False)
            optimizer_args['train_dataloader'] = train_dataloader
            optimizer_args['test_dataloader'] = test_dataloader
            if oD['name'] == 'CBO':
                time_start = time.time()
                CBO.train(model=hybrid_model, args=optimizer_args)
                experiment_time = time.time()-time_start
                hybrid_model, ckpt = CBO.get_best_parameters(hybrid_model, best_parameter_file)
                best_parameters = [p for p in hybrid_model.parameters()]
                if gS['plot_prediction']:
                    hybrid_model.t = t_train
                    hybrid_model.x0 =  x0_train
                    hybrid_model.trajectory_mode()
                    x_train_pred = hybrid_model()
                    hybrid_model.t = t_test
                    hybrid_model.x0 = x0_test
                    x_test_pred = hybrid_model()

            elif oD['name'] == 'ASM':
                flat_parameters, unravel_pytree = flatten_util.ravel_pytree(parameters)
                optimizer_args['unravel_pytree'] = unravel_pytree
                optimizer_args['model'] = datamodel
                optimizer_args['J_residual'] = J_residual
                optimizer_args['fmu_evaluator'] = fmu_evaluator
                time_start = time.time()
                if oD['method'] == 'Adam':
                    Adam = ASM.AdamOptim(eta=oD['adam_eta'], beta1=oD['adam_beta1'], beta2=oD['adam_beta2'], epsilon=oD['adam_eps'])
                    for i in range(optimizer_args['epochs']):
                        loss, grad = ASM.residual_wrapper(parameters=flat_parameters, args=optimizer_args)
                        flat_parameters = Adam.update(i+1, flat_parameters, grad)
                elif oD['method'] in ['CG', 'BFGS', 'SLSQP']:
                    minimize(ASM.residual_wrapper, flat_parameters, method=oD['method'], jac=True, args=optimizer_args, tol = oD['tol'], options={'maxiter':oD['epochs']})
                run_time = time.time() - time_start
                best_parameters = optimizer_args['best_parameters']
                best_loss = optimizer_args['best_loss']
                best_loss_test = optimizer_args['best_loss_test']
                x_train_pred = optimizer_args['best_pred']
                x_test_pred = optimizer_args['best_pred_test']
            else:
                logger.error(f'Unknown Optimizer: {oD["name"]}')
                x_test_pred = None
                x_train_pred = None
                best_parameters = None
                exit(1)

            ############################################################################
            ## PLOT RESULTS  ###########################################################
            ############################################################################

            if gS['plot_prediction']:
                result_plot_multi_dim(mD['name'], pD['name'], os.path.join(residual_directory, f'Best Prediction.png'),
                    t_train, x_train_pred, t_test, x_test_pred,
                    np.hstack((t_train, t_test)), np.vstack((x_ref_train, x_ref_test)))

            if gS['plot_loss']:
                build_plot(epochs=optimizer_args['epoch'],
                    model_name=mD['name'],
                    dataset_name=pD['name'],
                    plot_path=os.path.join(residual_directory, 'Loss.png'),
                    train_acc=optimizer_args['accuracies_train'],
                    test_acc=optimizer_args['accuracies_test'],
                    train_loss=optimizer_args['losses_train'],
                    test_loss=optimizer_args['losses_test']
                )

            if gS['plot_parameters']:
                visualise_wb(best_parameters, residual_directory, f'Best Parameters')

        if oD['trajectory_training']:
            ############################################################################
            ## TRAJECTORY PREPARATIONS  ################################################
            ############################################################################
            if 'NN' in mD['name']:
                if oD['name'] == 'CBO':
                    layers = [pD['inputs']] + mD['layers'] + [pD['outputs']]
                    datamodel = CustomMLP(layers)
                    hybrid_model = CBO.Hybrid_FMU(fmu_evaluator, datamodel, x0_train, t_train, mode='trajectory')
                    pointers = hybrid_model.pointers
                elif oD['name'] == 'ASM':
                    layers = mD['layers'] + [pD['outputs']]
                    datamodel, parameters, unjitted_nn = NN.create_nn(layers, x0_train)
                    hybrid_model = None
                    pointers = fmu_evaluator.pointers

                    dinput_dx_function_FMU = lambda p, x: jnp.array(jax.jacobian(unjitted_nn.apply, argnums=1)(p, x))
                    vectorized_dinput_dx_function_FMU = jax.jit(jax.vmap(dinput_dx_function_FMU, in_axes=(None, 0)))
                    dinput_dtheta_function_FMU = lambda p, x: jax.jacobian(unjitted_nn.apply, argnums=0)(p, x)
                    dinput_dtheta_function_FMU = jax.jit(dinput_dtheta_function_FMU)
                else:
                    hybrid_model = None
                    datamodel = None
                    logger.error(f'Unknown Model: {oD["name"]}')
                    exit(1)

                trajectory_directory, checkpoint_directory = create_results_subdirectories(results_directory=results_directory, trajectory=True, residual=False)
                if gS['save_parameters']:
                    if oD['name'] == 'CBO':
                        best_parameter_file = os.path.join(checkpoint_directory, gS['save_name'] + '.pt')
                        checkpoint_manager_save = None
                    elif oD['name'] == 'ASM':
                        checkpointer_save = orbax.checkpoint.PyTreeCheckpointer()
                        options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=5, create=True)
                        checkpoint_manager_save = orbax.checkpoint.CheckpointManager(checkpoint_directory, checkpointer_save, options)
                        best_parameter_file = None
                    else:
                        best_parameter_file = None
                        checkpoint_manager_save = None
                else:
                    best_parameter_file = None
                    checkpoint_manager_save = None

                ########################################################################
                #### LOAD PARAMETERS  ##################################################
                ########################################################################
                if oD['residual_training']:
                    # LOAD PARAMETERS FROM RESIDUAL TRAINING
                    if oD['name'] == 'CBO':
                        datamodel = CBO.load_parameters(hybrid_model, best_parameter_file)
                    elif oD['name'] == 'ASM':
                        checkpointer_load = orbax.checkpoint.PyTreeCheckpointer()
                        options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=5, create=True)
                        checkpoint_manager_load = orbax.checkpoint.CheckpointManager(checkpoint_directory, checkpointer_load, options)
                        step = checkpoint_manager_load.latest_step()
                        parameters = checkpoint_manager_load.restore(step)
                else:
                    if gS['load_parameters']:
                        # LOAD PARAMETERS FROM CHECKPOINT
                        load_directory = os.path.join(directory, gS['load_name'])
                        checkpointer_load = orbax.checkpoint.PyTreeCheckpointer()
                        options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=5, create=True)
                        checkpoint_manager_load = orbax.checkpoint.CheckpointManager(restore_directory, checkpointer_load, options)
                        step = checkpoint_manager_load.latest_step()
                        parameters = checkpoint_manager_load.restore(step)
                        if oD['name'] == 'CBO':
                            datamodel = CBO.load_jax_parameters(datamodel, parameters)
            else:
                logger.error(f'Unknown Model: {oD["name"]}')
                exit(1)

            ############################################################################
            ## PREPARE TRAINING ARGUMENTS  #############################################
            ############################################################################
            train_dataset = create_generic_dataset(torch.tensor(t_train), torch.tensor(x_ref_train))
            test_dataset = create_generic_dataset(torch.tensor(t_test), torch.tensor(x_ref_test))
            train_dataloader, test_dataloader = load_generic_dataloaders(train_dataset=train_dataset,
                                                        train_batch_size=oD['batch_size'] if oD['batch_size'] < len(train_dataset) else len(train_dataset),
                                                        test_dataset=test_dataset,
                                                        test_batch_size=oD['batch_size'] if oD['batch_size'] < len(test_dataset) else len(test_dataset),
                                                        shuffle=False)
            optimizer_args = {'problemDescription': pD,
                'modelDescription': mD,
                'optimizerDescription': oD,
                'generalSettings': gS,
                'epochs': oD['epochs'],
                'epoch': 0,
                'train_dataloader': train_dataloader,
                'test_dataloader': test_dataloader,
                'reference_data': reference_data,
                'losses_train' : [],
                'losses_train_res': [],
                'losses_test': [],
                'losses_test_res': [],
                'losses_batches': [], # only needed if we use batching
                'accuracies_train': [],
                'accuracies_test': [],
                'best_parameters': None,
                'best_loss': np.inf,
                'best_loss_test': np.inf,
                'best_pred': None,
                'best_pred_test': None,
                'best_parameter_file': best_parameter_file,
                'checkpoint_manager': checkpoint_manager_save,
                'logger': logger,
                'results_directory': trajectory_directory,
                'residual': False,
                'pointers': pointers # only needed for FMU
            }
            ############################################################################
            ## TRAINING  ###############################################################
            ############################################################################
            if oD['name'] == 'CBO':
                time_start = time.time()
                CBO.train(model=hybrid_model, args=optimizer_args)
                experiment_time = time.time()-time_start
                hybrid_model, ckpt = CBO.get_best_parameters(hybrid_model, best_parameter_file)
                best_parameters = [p for p in hybrid_model.parameters()]
                if gS['plot_prediction']:
                    hybrid_model.t = t_train
                    hybrid_model.x0 =  x0_train
                    hybrid_model.trajectory_mode()
                    x_train_pred = hybrid_model()
                    hybrid_model.t = t_test
                    hybrid_model.x0 = x0_test
                    x_test_pred = hybrid_model()
            elif oD['name'] == 'ASM':
                flat_parameters, unravel_pytree = flatten_util.ravel_pytree(parameters)
                inputs_train = []
                outputs_train = []
                for batch_idx, (input_train, output_train) in enumerate(optimizer_args['train_dataloader']):
                    inputs_train.append(input_train.detach().numpy())
                    outputs_train.append(output_train.detach().numpy())
                inputs_test = []
                outputs_test = []
                for batch_idx, (input_test, output_test) in enumerate(optimizer_args['test_dataloader']):
                    inputs_test.append(input_test.detach().numpy())
                    outputs_test.append(output_test.detach().numpy())
                optimizer_args['inputs_train'] = inputs_train
                optimizer_args['inputs_test'] = inputs_test
                optimizer_args['outputs_train'] = outputs_train
                optimizer_args['outputs_test'] = outputs_test
                optimizer_args['unravel_pytree'] = unravel_pytree
                optimizer_args['model'] = datamodel
                optimizer_args['fmu_evaluator'] = fmu_evaluator
                optimizer_args['vectorized_dinput_dx_function_FMU'] = vectorized_dinput_dx_function_FMU
                optimizer_args['dinput_dtheta_function_FMU'] = dinput_dtheta_function_FMU
                optimizer_args['vectorized_df_dtheta_function_FMU'] = NN.vectorized_df_dtheta_function_FMU

                time_start = time.time()
                if oD['method'] == 'Adam':
                    Adam = ASM.AdamOptim(eta=oD['adam_eta'], beta1=oD['adam_beta1'], beta2=oD['adam_beta2'], epsilon=oD['adam_eps'])
                    for i in range(optimizer_args['epochs']):
                        loss, grad = ASM.trajectory_wrapper(parameters=flat_parameters, args=optimizer_args)
                        flat_parameters = Adam.update(i+1, flat_parameters, grad)
                elif oD['method'] in ['CG', 'BFGS', 'SLSQP']:
                    minimize(ASM.trajectory_wrapper, flat_parameters, method=oD['method'], jac=True, args=optimizer_args, tol = oD['tol'], options={'maxiter':oD['epochs']})
                run_time = time.time() - time_start
                best_parameters = optimizer_args['best_parameters']
                best_loss = optimizer_args['best_loss']
                best_loss_test = optimizer_args['best_loss_test']
                x_train_pred = optimizer_args['best_pred']
                x_test_pred = optimizer_args['best_pred_test']
            else:
                logger.error(f'Unknown Optimizer: {oD["name"]}')
                best_parameters = None
                exit(1)

            ############################################################################
            ## PLOT RESULTS  ##########################################################
            ############################################################################
            if gS['plot_prediction']:
                result_plot_multi_dim(mD['name'], pD['name'], os.path.join(trajectory_directory, f'Best Prediction.png'),
                t_train, x_train_pred, t_test, x_test_pred, np.hstack((t_train, t_test)), np.vstack((x_ref_train, x_ref_test)))

            if gS['plot_loss']:
                build_plot(epochs=len(optimizer_args['losses_train']),
                    model_name=mD['name'],
                    dataset_name=pD['name'],
                    plot_path=os.path.join(trajectory_directory, 'Loss.png'),
                    train_acc=optimizer_args['accuracies_train'],
                    test_acc=optimizer_args['accuracies_test'],
                    train_loss=optimizer_args['losses_train'],
                    test_loss=optimizer_args['losses_test']
                )

            if gS['plot_parameters']:
                visualise_wb(best_parameters, trajectory_directory, f'Best Parameters')

        if 'NN' in mD['name']:
            # NN_FMU_NAME = f'{mD["name"]}_{pD["name"]}_{oD["name"]}'
            NN_FMU_NAME = pD['model_fmu_name']
            path  = os.path.join(directory, NN_FMU_NAME, "build", NN_FMU_NAME + ".fmu")
            targetDirPath = os.path.join(directory, NN_FMU_NAME)
            # if oD['name'] == 'ASM':
                # best_parameters, _ = flatten_util.ravel_pytree(best_parameters)
            create_NN_FMU(targetDirPath, NN_FMU_NAME, best_parameters, pD['inputs'], pD['outputs'])
            fmu_evaluator_NN = FMUEvaluator(path, 0, 1)

            nn_fmu_eval_function = fmu_evaluator_NN.evaluate_fmu_NN

            if oD['name'] == 'CBO':
                hybrid_model.t = t_train
                hybrid_model.x0 =  x0_train
                hybrid_model.trajectory_mode()
                x_train_pred = hybrid_model()
                hybrid_model.t = t_test
                hybrid_model.x0 = x0_test
                x_test_pred = hybrid_model()

                hybrid_model.augment_model_function = nn_fmu_eval_function
                hybrid_model.t = t_train
                hybrid_model.x0 =  x0_train
                hybrid_model.trajectory_mode()
                x_train_fmu = hybrid_model()
                hybrid_model.t = t_test
                hybrid_model.x0 = x0_test
                x_test_fmu = hybrid_model()
            elif oD['name'] == 'ASM':
                fmu_evaluator.training = False
                x_train_fmu = fmu_evaluator.euler(x0_train, t_train, model=nn_fmu_eval_function, model_parameters=None, )
                fmu_evaluator.reset_fmu(pD['integration_parameters']['end'], pD['integration_parameters']['end']*(test_factor+1))
                x_test_fmu = fmu_evaluator.euler(x0_test, t_test, model=nn_fmu_eval_function, model_parameters=None)
                fmu_evaluator.reset_fmu(pD['integration_parameters']['start'], pD['integration_parameters']['end'])
            result_plot_multi_dim(mD['name'], pD['name'], os.path.join(results_directory, f'Loaded FMU Prediction.png'),
                t_train, x_train_fmu, t_test, x_test_fmu, np.hstack((t_train, t_test)), np.vstack((x_train_pred, x_test_pred)))


    ####################################################################################
    #### NO FMU  #######################################################################
    ####################################################################################
    else:
        # 1. Create Reference Solution
        # A reference solution is needed for all problems, models and optimizers
        x0_train = []
        for variable_name in pD['initial_conditions'].keys():
            x0_train.append(pD['initial_conditions'][variable_name])
        x0_train = np.array(x0_train)
        logger.info('Creating Reference Solution')
        namespace = namespaces[pD['name']]
        ode = namespace.ode
        ode_res = namespace.ode_res
        ode_hybrid = namespace.ode_hybrid
        t_train, x_ref_train, t_test, x_ref_test, x0_test = create_reference_solution(
            pD['integration_parameters']['start'],
            pD['integration_parameters']['end'],
            pD['integration_parameters']['steps'],
            x0_train,
            pD['variables'],
            ode_integrator=euler,
            ode=ode,
            test_factor=test_factor)

        reference_data = {
            't_train': t_train,
            'x_ref_train': x_ref_train,
            't_test': t_test,
            'x_ref_test': x_ref_test
        }

        if 'NN' in mD['name']:
            if oD['name'] == 'CBO':
                layers = [pD['inputs']] + mD['layers'] + [pD['outputs']]
                datamodel = CustomMLP(layers)
                hybrid_model = CBO.Hybrid_Python(pD['variables'], datamodel, x0_train, t_train, ode_hybrid, mode='residual' if oD['residual_training'] else 'trajectory')
            elif oD['name'] == 'ASM':
                layers = mD['layers'] + [pD['outputs']]
                datamodel, parameters, _ = NN.create_nn(layers, x0_train)
                hybrid_model = None
                @jit
                def J_residual(inputs, outputs, parameters):
                    def squared_error(input, output):
                        pred = datamodel(parameters, input)
                        return (output-pred)**2, pred
                    J, pred = jax.vmap(squared_error)(inputs, outputs)
                    return jnp.mean(J, axis=0)[0], pred
                df_dtheta_function = lambda x, t, phi, theta: jax.jacobian(ode_hybrid, argnums=3)(x, t, phi, theta, datamodel)
                vectorized_df_dtheta_function = jit(jax.vmap(df_dtheta_function, in_axes=(0, 0, None, None)))
                df_dt_function = lambda x, t, phi, theta: jax.jacobian(ode_hybrid, argnums=2)(x, t, phi, theta, datamodel)
                df_dt_function = jit(df_dt_function)
            else:
                hybrid_model = None
                datamodel = None
                logger.error(f'Unknown Model: {oD["name"]}')
                exit(1)

            if gS['load_parameters']:
                load_directory = os.path.join(directory, gS['load_name'])
                checkpointer_load = orbax.checkpoint.PyTreeCheckpointer()
                options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=5, create=True)
                checkpoint_manager_load = orbax.checkpoint.CheckpointManager(restore_directory, checkpointer_load, options)
                step = checkpoint_manager_load.latest_step()
                parameters = checkpoint_manager_load.restore(step)
                if oD['name'] == 'CBO':
                    datamodel = CBO.load_jax_parameters(datamodel, parameters)
        else:
            logger.error(f'Unknown Model: {oD["name"]}')
            exit(1)

        if oD['residual_training']:
            residual_directory, checkpoint_directory = create_results_subdirectories(results_directory=results_directory, trajectory=False, residual=True)
            if gS['save_parameters']:
                if oD['name'] == 'CBO':
                    best_parameter_file = os.path.join(checkpoint_directory, gS['save_name'] + '.pt')
                    checkpoint_manager_save = None
                elif oD['name'] == 'ASM':
                    checkpointer_save = orbax.checkpoint.PyTreeCheckpointer()
                    options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=5, create=True)
                    checkpoint_manager_save = orbax.checkpoint.CheckpointManager(checkpoint_directory, checkpointer_save, options)
                    best_parameter_file = None
                else:
                    best_parameter_file = None
                    checkpoint_manager_save = None
            else:
                best_parameter_file = None
                checkpoint_manager_save = None
            x_train_res, y_train_res, x_test_res, y_test_res = create_residual_reference_solution(t_train, x_ref_train, t_test, x_ref_test, pD['variables'], ode_res)

            # Arguments for ASM and CBO
            optimizer_args = {'problemDescription': pD,
                'modelDescription': mD,
                'optimizerDescription': oD,
                'generalSettings': gS,
                'epochs': oD['epochs_res'],
                'epoch': 0,
                'reference_data': reference_data,
                'losses_train' : [],
                'losses_train_res': [],
                'losses_test': [],
                'losses_test_res': [],
                'losses_batches': [], # only needed if we use batching
                'accuracies_train': [],
                'accuracies_test': [],
                'best_parameters': None,
                'best_loss': np.inf,
                'best_loss_test': np.inf,
                'best_pred': None,
                'best_pred_test': None,
                'best_parameter_file': best_parameter_file,
                'checkpoint_manager': checkpoint_manager_save,
                'logger': logger,
                'results_directory': residual_directory,
                'residual': True,
                'pointers': None, # only needed for FMU
                'ode_hybrid': ode_hybrid # only needed for Python only implementation
            }

            # if pD['type'] == 'classification':
            #     optimizer_args['accuracies_train'] = []
            #     optimizer_args['accuracies_test'] = []

            results_dict = {}

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

            if oD['name'] == 'CBO':
                CBO.train(model=hybrid_model, args=optimizer_args)
                experiment_time = time.time()-time_start
                datamodel, ckpt = CBO.get_best_parameters(hybrid_model, best_parameter_file)
                best_parameters = [p for p in datamodel.parameters()]

                if gS['plot_prediction']:
                    datamodel.t = t_train
                    datamodel.x0 =  x0_train
                    datamodel.trajectory_mode()
                    x_train_pred = datamodel()
                    datamodel.t = t_test
                    datamodel.x0 = x0_test
                    x_test_pred = datamodel()

            elif oD['name'] == 'ASM':
                flat_parameters, unravel_pytree = flatten_util.ravel_pytree(parameters)
                optimizer_args['unravel_pytree'] = unravel_pytree
                optimizer_args['model'] = datamodel
                optimizer_args['J_residual'] = J_residual


                time_start = time.time()
                if oD['method'] == 'Adam':
                    Adam = ASM.AdamOptim(eta=oD['adam_eta'], beta1=oD['adam_beta1'], beta2=oD['adam_beta2'], epsilon=oD['adam_eps'])
                    for i in range(optimizer_args['epochs']):
                        loss, grad = ASM.residual_wrapper(parameters=flat_parameters, args=optimizer_args)
                        flat_parameters = Adam.update(i+1, flat_parameters, grad)
                elif oD['method'] in ['CG', 'BFGS', 'SLSQP']:
                    minimize(ASM.residual_wrapper, flat_parameters, method=oD['method'], jac=True, args=optimizer_args, tol = oD['tol'], options={'maxiter':oD['epochs']})
                run_time = time.time() - time_start

                best_parameters = optimizer_args['best_parameters']
                best_loss = optimizer_args['best_loss']
                best_loss_test = optimizer_args['best_loss_test']
                x_train_pred = optimizer_args['best_pred']
                x_test_pred = optimizer_args['best_pred_test']
            else:
                logger.error(f'Unknown Optimizer: {oD["name"]}')
                best_parameters = None
                exit(1)


            if gS['plot_prediction']:
                result_plot_multi_dim(mD['name'], pD['name'], os.path.join(results_directory, f'Best Prediction (Residual).png'),
                    t_train, x_train_pred, t_test, x_test_pred,
                    np.hstack((t_train, t_test)), np.vstack((x_ref_train, x_ref_test)))

            if gS['plot_loss']:
                build_plot(epochs=len(optimizer_args['losses_train']),
                    model_name=mD['name'],
                    dataset_name=pD['name'],
                    plot_path=os.path.join(results_directory, 'Loss (Residual).png'),
                    train_acc=optimizer_args['accuracies_train'],
                    test_acc=optimizer_args['accuracies_test'],
                    train_loss=optimizer_args['losses_train'],
                    test_loss=optimizer_args['losses_test']
                )

            if gS['plot_parameters']:
                visualise_wb(best_parameters, results_directory, f'Best Parameters (Residual)')


        # NOW TRAJECTORY TRAINING. ALWAYS HAPPENS
        # CONVERT THE REFERENCE DATA TO A DATASET

        if oD['trajectory_training']:
            trajectory_directory, checkpoint_directory = create_results_subdirectories(results_directory=results_directory, trajectory=True, residual=False)
            if gS['save_parameters']:
                if oD['name'] == 'CBO':
                    best_parameter_file = os.path.join(checkpoint_directory, gS['save_name'] + '.pt')
                    checkpoint_manager_save = None
                elif oD['name'] == 'ASM':
                    checkpointer_save = orbax.checkpoint.PyTreeCheckpointer()
                    options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=5, create=True)
                    checkpoint_manager_save = orbax.checkpoint.CheckpointManager(checkpoint_directory, checkpointer_save, options)
                    best_parameter_file = None
                else:
                    best_parameter_file = None
                    checkpoint_manager_save = None
            else:
                best_parameter_file = None
                checkpoint_manager_save = None
            # Arguments for ASM and CBO
            optimizer_args = {'problemDescription': pD,
                'modelDescription': mD,
                'optimizerDescription': oD,
                'generalSettings': gS,
                'epochs': oD['epochs'],
                'epoch': 0,
                'reference_data': reference_data,
                'losses_train' : [],
                'losses_train_res': [],
                'losses_test': [],
                'losses_test_res': [],
                'losses_batches': [], # only needed if we use batching
                'accuracies_train': [],
                'accuracies_test': [],
                'best_parameters': None,
                'best_loss': np.inf,
                'best_loss_test': np.inf,
                'best_pred': None,
                'best_pred_test': None,
                'best_parameter_file': best_parameter_file,
                'checkpoint_manager': checkpoint_manager_save,
                'logger': logger,
                'results_directory': trajectory_directory,
                'residual': False,
                'pointers': None, # only needed for FMU
                'ode_hybrid': ode_hybrid # only needed for Python only implementation
            }

            train_dataset = create_generic_dataset(torch.tensor(t_train), torch.tensor(x_ref_train))
            test_dataset = create_generic_dataset(torch.tensor(t_test), torch.tensor(x_ref_test))
            train_dataloader, test_dataloader = load_generic_dataloaders(train_dataset=train_dataset,
                                                                            train_batch_size=oD['batch_size'] if oD['batch_size'] < len(train_dataset) else len(train_dataset),
                                                                            test_dataset=test_dataset,
                                                                            test_batch_size=oD['batch_size'] if oD['batch_size'] < len(test_dataset) else len(test_dataset),
                                                                            shuffle=False)
            optimizer_args['train_dataloader'] = train_dataloader
            optimizer_args['test_dataloader'] = test_dataloader

            time_start = time.time()

            if oD['name'] == 'CBO':
                CBO.train(model=hybrid_model, args=optimizer_args)

                experiment_time = time.time()-time_start
                hybrid_model, ckpt = CBO.get_best_parameters(hybrid_model, best_parameter_file)
                if gS['plot_prediction']:
                    hybrid_model.t = t_train
                    hybrid_model.x0 =  x0_train
                    hybrid_model.trajectory_mode()
                    x_train_pred = hybrid_model()
                    hybrid_model.t = t_test
                    hybrid_model.x0 = x0_test
                    x_test_pred = hybrid_model()

            elif oD['name'] == 'ASM':
                flat_parameters, unravel_pytree = flatten_util.ravel_pytree(parameters)
                optimizer_args['unravel_pytree'] = unravel_pytree
                optimizer_args['model'] = datamodel
                optimizer_args['vectorized_df_dtheta_function'] = vectorized_df_dtheta_function
                optimizer_args['df_dt_function'] = df_dt_function
                time_start = time.time()
                inputs_train = []
                outputs_train = []
                for batch_idx, (input_train, output_train) in enumerate(optimizer_args['train_dataloader']):
                    inputs_train.append(input_train.detach().numpy())
                    outputs_train.append(output_train.detach().numpy())
                inputs_test = []
                outputs_test = []
                for batch_idx, (input_test, output_test) in enumerate(optimizer_args['test_dataloader']):
                    inputs_test.append(input_test.detach().numpy())
                    outputs_test.append(output_test.detach().numpy())
                optimizer_args['inputs_train'] = inputs_train
                optimizer_args['inputs_test'] = inputs_test
                optimizer_args['outputs_train'] = outputs_train
                optimizer_args['outputs_test'] = outputs_test
                optimizer_args['loss_batches'] = []

                if oD['method'] == 'Adam':
                    Adam = ASM.AdamOptim(eta=oD['adam_eta'], beta1=oD['adam_beta1'], beta2=oD['adam_beta2'], epsilon=oD['adam_eps'])
                    for i in range(oD['epochs_res']):
                        loss, grad = ASM.trajectory_wrapper(parameters=flat_parameters, args=optimizer_args)
                        flat_parameters = Adam.update(i+1, flat_parameters, grad)
                elif oD['method'] in ['CG', 'BFGS', 'SLSQP']:
                    minimize(ASM.trajectory_wrapper, flat_parameters, method=oD['method'], jac=True, args=optimizer_args, options={'maxiter':oD['epochs'], 'tol': oD['tol']})

                run_time = time.time() - time_start

                best_parameters = optimizer_args['best_parameters']
                best_loss = optimizer_args['best_loss']
                best_loss_test = optimizer_args['best_loss_test']
                x_train_pred = optimizer_args['best_pred']
                x_test_pred = optimizer_args['best_pred_test']
            else:
                logger.error(f'Unknown Optimizer: {oD["name"]}')
                exit(1)


            if gS['plot_prediction']:
                result_plot_multi_dim(mD['name'], pD['name'], os.path.join(results_directory, f'Best Prediction (Trajecotry).png'),
                t_train, x_train_pred, t_test, x_test_pred, np.hstack((t_train, t_test)), np.vstack((x_ref_train, x_ref_test)))

            if gS['plot_loss']:
                build_plot(epochs=len(optimizer_args['losses_train']),
                    model_name=mD['name'],
                    dataset_name=pD['name'],
                    plot_path=os.path.join(results_directory, 'Loss (Trajectory).png'),
                    train_acc=optimizer_args['accuracies_train'],
                    test_acc=optimizer_args['accuracies_test'],
                    train_loss=optimizer_args['losses_train'],
                    test_loss=optimizer_args['losses_test']
                )

            if gS['plot_parameters']:
                visualise_wb(best_parameters, results_directory, f'Best Parameters (Trajectory)')





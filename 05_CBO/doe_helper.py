from pyDOE2 import fullfact, lhs
import yaml
import argparse
import subprocess
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import os
from utils import create_results_directory
import logging

def read_doe_file(file):
    pass

def create_experiments(variable_arguments:dict, fixed_arguments:dict, method='fullfact'):
    levels = [len(val) for val in variable_arguments.values()]
    if method == 'fullfact':
        doe = fullfact(levels)
    # elif method == 'lhs':
    #     doe = lhs(levels)
    else:
        print('Method not supported, using fullfact')
        doe = fullfact(levels)
    experiments = []
    for experiment in doe:
        experiment_dict = {}
        for i, key in enumerate(variable_arguments.keys()):
            experiment_dict[key] = variable_arguments[key][int(experiment[i])]
        experiment_dict = {**experiment_dict, **fixed_arguments}
        experiments.append(experiment_dict)
    return tuple(experiments)

def create_argument_list(experiment):
    ks = deepcopy(list(experiment.keys()))
    vs = deepcopy(list(experiment.values()))
    argument_list = []
    for k, v in zip(ks, vs):
        # experiment.pop(k)
        argument_list += [f'--{k}', f'{v}']
    return argument_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--script_file', type=str, default='11_CBO_single_experiment.py', help='The script for the design of experiments')
    parser.add_argument('--argument_file', type=str, default='doe_arguments.yaml',
                        help='The yaml file which contains the DoE arguments for the script')
    parser.add_argument('--results_directory_name', type=str, default='DoE_VdP_Particles (save)',
                        help='The yaml file which contains the DoE arguments for the script')
    parser.add_argument('--results_file', type=str, default='doe_results.yaml',
                        help='The yaml file which contains the DoE arguments for the script')
    parser.add_argument('--n_runs', type=int, default=3)
    args = parser.parse_args()


    path = os.path.abspath(__file__)
    directory = os.path.sep.join(path.split(os.path.sep)[:-1])
    results_directory = create_results_directory(directory=directory, results_directory_name=args.results_directory_name)

    log_file = os.path.join(results_directory, 'DoE.log')
    logging.basicConfig(filename=log_file, encoding='utf-8', level=logging.DEBUG)
    logger = logging.getLogger('DoE')

    doe_argument_file = os.path.join(directory, 'VdP_CBO_DoE.yaml')
    script_file = os.path.join(directory, args.script_file)

    # variable_arguments = {
    #     'n_particles': [100, 20],
    #     'sigma': [0.1, 0.4],
    # }
    # fixed_arguments = {
    #     'epochs': 10,
    #     'batch_size': 100,
    #     'results_file': args.results_file
    # # }
    # arguments = {'fixed_arguments': fixed_arguments, 'variable_arguments': variable_arguments}

    # with open('VdP_CBO_DoE.yaml', 'w') as file:
    #     yaml.dump(arguments, file)

    with open(doe_argument_file, 'r') as file:
        doc_generator = yaml.safe_load_all(file)
        for arguments in doc_generator:
            pass

    fixed_arguments = arguments['fixed_arguments']
    variable_arguments = arguments['variable_arguments']
    fullfact_experiments = create_experiments(variable_arguments, fixed_arguments, method='fullfact')

    with open(os.path.join(results_directory, 'experiment_list.yaml'), 'w') as file:
        experiments = {f'Experiment {n_exp}': experiment for n_exp, experiment in enumerate(fullfact_experiments)}
        yaml.dump(experiments, file)

    all_results = [None]*len(fullfact_experiments)
    averaged_results = [None]*len(fullfact_experiments)

    for run in range(1, args.n_runs+1):
        logger.info(f'Starting run {run}')
        run_directory = os.path.join(results_directory, f'run_{run}')
        os.mkdir(run_directory)

        procs = []
        for n_exp, exp in enumerate(fullfact_experiments):
            logger.info(f'Starting Experiment {n_exp}')

            experiment_directory = os.path.join(run_directory, f'experiment_{n_exp}')
            os.mkdir(experiment_directory)
            experiment_file = os.path.join(experiment_directory, args.results_file)
            exp['results_file'] = experiment_file
            exp['results_directory'] = experiment_directory

            # RUN THE SCRIPT AND COllECT THE RESULTS
            fullfact_argument_list = create_argument_list(exp)
            # result = subprocess.run(['python', script_file] + fullfact_argument_list, capture_output=True)
            procs.append(subprocess.Popen(['python', script_file] + fullfact_argument_list))

        for p in procs:
            p.wait()

        for n_exp, exp in enumerate(fullfact_experiments):
            experiment_directory = os.path.join(run_directory, f'experiment_{n_exp}')
            experiment_file = os.path.join(experiment_directory, args.results_file)
            with open(experiment_file, 'r') as file:
                doc_generator = yaml.safe_load_all(file)
                for results in doc_generator:
                    pass

            # PARSE THE RESULTS FOR NUMERICAL VALUES OR LIST/TUPLE OF NUM. VALUES
            exp_result = {}
            for k, v in results.items():
                if type(v) is int or type(v) is float:
                    exp_result[k] = v
                else:
                    if type(v) is list or type(v) is tuple:
                        exp_result[k] = v
                        for elem in v:
                            if type(elem) is not int and type(elem) is not float:
                                exp_result.pop(k)
                                print(f'List/Tuple in {k} contains non numeric value. Ignoring {k}')
                                break
                    else:
                        print(f'{k} contains non numeric value. Ignoring {k}')

            # SAVE THE PARSED RESULTS FOR EACH EXPERIMENT AND RUN AND CALCULATE AVERAGES
            if run == 1:
                all_results[n_exp] = {}
                averaged_results[n_exp] = {}
                for k, v in exp_result.items():
                    all_results[n_exp][k] = [v]
                    averaged_results[n_exp][k] = v
            else:
                for k, v in exp_result.items():
                    all_results[n_exp][k].append(v)
                    a = np.asarray(all_results[n_exp][k]).mean(axis=0)
                    averaged_results[n_exp][k] = list(a) if type(a) is np.ndarray else a

        # UPDATE PLOTS
        for result_parameter, v in averaged_results[0].items():
            if type(v) is list or type(v) is tuple:
                fig, ax = plt.subplots(1,1)
                for exp_index, averaged_result in enumerate(averaged_results):
                    value = averaged_result[result_parameter]
                    x = range(len(value))
                    ax.plot(x, value, label= f'Exp.: {exp_index}')
                    ax.legend()
                    ax.set_ylabel(result_parameter)
                    ax.set_title(result_parameter + f' - ({run} runs)')
                plt.savefig(os.path.join(results_directory, f'{result_parameter}.png'))
                plt.close()
            elif type(v) is np.int64 or type(v) is np.float64 or type(v) is int or type(v) is float:
                fig, ax = plt.subplots(1, 1)
                for exp_index, averaged_result in enumerate(averaged_results):
                    value = averaged_result[result_parameter]
                    ax.bar(exp_index, value, label= f'Exp.: {exp_index}')
                    ax.set_ylabel(result_parameter)
                    ax.set_title(result_parameter + f' - ({run} runs)')
                    ax.legend()
                plt.savefig(os.path.join(results_directory, f'{result_parameter}.png'))
                plt.close()




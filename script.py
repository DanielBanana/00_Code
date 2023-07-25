import argparse
import sys
import yaml
import numpy as np
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--n_particles', type=int, default=10)
    parser.add_argument('--sigma', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--results_file', type=str, default='doe_results.yaml')
    parser.add_argument('--results_directory', type=str)
    args = parser.parse_args()

    rnd = np.random.normal()


    loss_history = [40, 25, 10, 6, 1, 0.1]
    for i, l in enumerate(loss_history):
        loss_history[i] += rnd
    time = 15.3 + 10*rnd
    string = 'string'
    list_w_string = [10, 20, 'str', 10]

    results_dict = {
        'loss_history': loss_history,
        'time': time,
        # 'string': string,
        # 'list_w_string': list_w_string
    }

    with open(os.path.join(args.results_directory, args.results_file), 'w') as file:
        yaml.dump(results_dict, file)

    exit(0)
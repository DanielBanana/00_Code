import os
import sys
import argparse

def get_cpp_parameter(param_name, param_value, param_type, param_shape):
    cpp_parameter = param_type
    cpp_parameter = " ".join([cpp_parameter, param_name])
    for d in param_shape:
        cpp_parameter += f"[{d}]"
    param_value = f" = {param_value};".replace("[", "{").replace("]", "}")
    cpp_parameter += param_value
    return cpp_parameter


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # # PROBLEM SETUP
    # parser.add_argument('--dataset', type=str, default='VdP', help='dataset to use',
    #                     choices=list(DATASETS.keys()))



    nn_parameters = {"weights1": {"param_values": [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], "param_type": "double", "param_shape": [15, 1]},
                     "bias1": {"param_values": [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], "param_type": "double", "param_shape": [15]},
                     "weights2": {"param_values": [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], "param_type": "double", "param_shape": [1, 15]},
                     "bias2": {"param_values": [1], "param_type": "double", "param_shape": [1]}}

    cpp_parameter_strings = []
    for name, values in nn_parameters.items():
        string = get_cpp_parameter(name, values["param_values"], values["param_type"], values["param_shape"])
        cpp_parameter_strings.append(string)
        print(string)

    path = os.path.abspath(__file__)
    directory = os.path.sep.join(path.split(os.path.sep)[:-1])
    filename = "parameters.hpp"

    with open(os.path.join(directory,filename), 'w') as file:
        for string in cpp_parameter_strings:
            file.write(string+'\n')


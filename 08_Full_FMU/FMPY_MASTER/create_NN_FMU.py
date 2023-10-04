import jax
import numpy as np
import jax.numpy as jnp
import flax
from flax.core import unfreeze
from jax import random, numpy as jnp
from flax import linen as nn
from typing import Sequence
import os
import platform
import subprocess
import xml.etree.ElementTree as ET

# from create_parameters_h import create_params_header
# from create_cpp_code import create_cpp_code
# from create_modelDescription import create_modelDescription
# from build_fmu import build_fmu


def create_NN_FMU(targetDirPath, modelName, params, n_inputs, n_outputs):
    if os.path.exists(os.path.join(targetDirPath, "src", "weights_biases.h")):
        print("NN at Path already created. Just the parameters will get replaced")
        create_params_header(targetDirPath=targetDirPath, params=params)
        build_fmu(targetDirPath=targetDirPath, modelName=modelName)
    else:
        guid = create_modelDescription(targetDirPath=targetDirPath, n_inputs=n_inputs, n_outputs=n_outputs)
        create_params_header(targetDirPath=targetDirPath, params=params)
        create_cpp_code(targetDirPath=targetDirPath,
                        modelName=modelName,
                        n_inputs=n_inputs,
                        n_outputs=n_outputs,
                        guid=guid)
        build_fmu(targetDirPath=targetDirPath, modelName=modelName)

def build_fmu(targetDirPath, modelName):

    # def testBuildFMU(self):
    # """Runs a cmake-based compilation of the generated FMU to check if the code compiles.
    # """

    # generate path to /build subdir
    buildDir = os.path.join(targetDirPath, "build")
    binDir = os.path.join(targetDirPath, "bin/release")

    print("We are now building the FMU.")
    try:

        # Different script handling based on platform
        if platform.system() == "Windows":

            # call batch file to build the FMI library
            pipe = subprocess.Popen(["build_VC_x64.bat"], shell=True, creationflags=subprocess.CREATE_NEW_CONSOLE, cwd = buildDir, stdout = subprocess.PIPE, stderr = subprocess.PIPE)
            # retrieve output and error messages
            outputMsg, errorMsg = pipe.communicate()
            # get return code
            rc = pipe.returncode

            # if return code is different from 0, print the error message
            if rc != 0:
                print(str(outputMsg) + "\n" + str(errorMsg))
                raise RuntimeError("Error during compilation of FMU.")

            print("Compiled FMU successfully")

            # call batch file to build the FMI library
            pipe = subprocess.Popen(["deploy.bat"], shell=True, creationflags=subprocess.CREATE_NEW_CONSOLE, cwd = buildDir, stdout = subprocess.PIPE, stderr = subprocess.PIPE)
            # retrieve output and error messages
            outputMsg, errorMsg = pipe.communicate()
            # get return code
            rc = pipe.returncode

            if rc != 0:
                print(str(outputMsg) + "\n" + str(errorMsg))
                raise RuntimeError("Error during compilation of FMU")

            print("Successfully created {}".format(modelName + ".fmu")	)

        else:
            # shell file execution for Mac & Linux
            pipe = subprocess.Popen(["bash", './build.sh'], cwd = buildDir, stdout = subprocess.PIPE, stderr = subprocess.PIPE)
            outputMsg,errorMsg = pipe.communicate()
            rc = pipe.returncode

            if rc != 0:
                print(errorMsg)
                raise RuntimeError("Error during compilation of FMU")

            print("Compiled FMU successfully")

            # Deployment

            # shell file execution for Mac & Linux
            deploy = subprocess.Popen(["bash", './deploy.sh'], cwd = buildDir, stdout = subprocess.PIPE, stderr = subprocess.PIPE)
            outputMsg,errorMsg = deploy.communicate()
            dc = deploy.returncode

            if dc != 0:
                print(errorMsg)
                raise RuntimeError("Error during assembly of FMU")

            print("Successfully created {}".format(modelName + ".fmu")	)

    except Exception as e:
        print(str(e))
        print("Error building FMU.")
        raise

def create_modelDescription(targetDirPath, n_inputs, n_outputs):

    file_name = "modelDescription.xml"
    data_path = os.path.join(targetDirPath, "data")
    file_path = os.path.join(data_path, file_name)
    file_path_tmp = os.path.join(data_path, 'tmp.xml')

    with open(file_path, 'rb') as xml_file:

        tree = ET.parse(xml_file)
        root = tree.getroot()

        guid = root.attrib['guid']

        # Function to add a ScalarVariable with specified attributes
        def add_scalar_variable(parent, name, description, value_reference, variability, causality, initial=None):

            attribs = {
                'name': name,
                'description': description,
                'valueReference': str(value_reference),
                'variability': variability,
                'causality': causality,
            }
            if initial :
                attribs['initial'] = initial

            scalar_variable = ET.SubElement(parent, 'ScalarVariable', attribs)

            if causality == 'input':
                real = ET.SubElement(scalar_variable, 'Real', {'start': "1.0"})
            if causality == 'output' and (initial == 'exact' or initial == 'approx'):
                real = ET.SubElement(scalar_variable, 'Real', {'start': "0.0"})

            return scalar_variable

        n_variables = n_inputs + n_outputs

        # Add ScalarVariables with causality "input"
        input_variables = [
            (f"FMI_INPUT_{i}",f"Input Description {i}", i, "continuous", "input") for i in range(1, n_inputs+1)
        ]

        # Add ScalarVariables with causality "output"
        output_variables = [
            (f"FMI_OUTPUT_{j-n_inputs}", f"Output Description {j-n_inputs}", j, "continuous", "output", "exact") for j in range(n_inputs+1, n_variables+1)
        ]

        # Find the ModelVariables element
        model_variables = root.find('ModelVariables')

        # Find the ModelStructure element
        model_structure = root.find('ModelStructure')

        # Add ScalarVariables to ModelVariables
        for name, description, value_reference, variability, causality in input_variables:
            add_scalar_variable(model_variables, name, description, value_reference, variability, causality)

        for name, description, value_reference, variability, causality, initial in output_variables:
            add_scalar_variable(model_variables, name, description, value_reference, variability, causality, initial)

        # Add dependencies to ModelStructure for all outputs on all inputs
        if model_structure is not None:
            outputs = model_structure.find('Outputs')
            if outputs is None:
                outputs = ET.SubElement(model_structure, 'Outputs')

            for output_variable in output_variables:
                index = output_variable[2]
                dependencies = ' '.join(str(variable[2]) for variable in input_variables)
                unknown = ET.SubElement(outputs, 'Unknown', attrib={'index': str(index), 'dependencies': dependencies})

        # Save the modified XML to a new file
        tree.write(file_path_tmp, encoding='utf-8', xml_declaration=True, method='xml', short_empty_elements=False)

    os.remove(file_path)
    os.rename(file_path_tmp, file_path)

    print(f"Modified XML file saved as '{file_name}'")

    return guid

def extract_params(pytree):
    flat_params = flax.traverse_util.flatten_dict(pytree)
    flat_params = {k: jnp.array(v) for k, v in flat_params.items()}
    return flat_params

def create_params_header(targetDirPath, params):

    file_path = os.path.join(targetDirPath, "src", "weights_biases.h")

    flat_params = extract_params(params)

    # Generate C++ header file content
    cpp_header_content = '#ifndef WEIGHTS_BIASES_H\n'
    cpp_header_content += '#define WEIGHTS_BIASES_H\n\n'
    cpp_header_content += '#include <vector>\n\n'
    cpp_header_content += 'namespace NeuralNetworkParams {\n'

    layer_weights = {}  # Store layer weights as nested vectors
    layer_biases = {}   # Store layer biases as vectors
    layer_sizes = []

    for key, param in flat_params.items():
        layer_name = key[1]  # Extract the layer name
        param_name = key[-1]  # Extract the parameter name ('kernel' or 'bias')

        if param_name == 'kernel':
            if layer_name not in layer_weights:
                # Convert the parameter values to a nested vector of doubles
                param_values = param.tolist()
                layer_weights[layer_name] = param_values
                layer_sizes.append(str(len(layer_weights[layer_name])))
        elif param_name == 'bias':
            if layer_name not in layer_biases:
                # Convert the parameter values to a vector of doubles
                param_values = param.tolist()
                layer_biases[layer_name] = param_values
    layer_sizes.append(str(len(layer_weights[layer_name][0])))

    # Write weights
    cpp_header_content += '\tconst std::vector<std::vector<std::vector<double>>> weights = {\n'
    for layer_name, weights in layer_weights.items():
        cpp_header_content += '\t\t// ' + layer_name + ' weights\n'
        cpp_header_content += '\t\t{\n'
        for weight_matrix in weights:
            cpp_header_content += '\t\t\t{'
            cpp_header_content += ', '.join(map(str, weight_matrix))
            cpp_header_content += '},\n'
        cpp_header_content += '\t\t},\n'
    cpp_header_content += '\t};\n'

    # Write biases
    cpp_header_content += '\n\tconst std::vector<std::vector<double>> biases = {\n'
    for layer_name, biases in layer_biases.items():
        cpp_header_content += '\t\t// ' + layer_name + ' biases\n'
        cpp_header_content += '\t\t{'
        cpp_header_content += ', '.join(map(str, biases))
        cpp_header_content += '},\n'
    cpp_header_content += '\t};\n'

    layer_sizes = ', '.join(layer_sizes)

    cpp_header_content += '}\n'

    cpp_header_content += 'std::vector<int> layer_sizes = {{{LAYER_SIZES}}};\n'.format(LAYER_SIZES = layer_sizes)

    cpp_header_content += '\n#endif\n'


    # Write the C++ header file
    with open(file_path, 'w') as header_file:
        header_file.write(cpp_header_content)

def create_cpp_code(targetDirPath, modelName, n_inputs, n_outputs, guid):

    cpp_path = os.path.join(targetDirPath,"src",modelName+".cpp")
    h_path = os.path.join(targetDirPath,"src",modelName+".h")
    h_path_tmp = os.path.join(targetDirPath,"src",modelName+"_tmp.h")

    h_string = """
    private:
		std::vector<std::vector<std::vector<double>>> weights;
		std::vector<std::vector<double>> biases;
		std::vector<std::vector<double>> activations;

	public:
		void neural_network(const std::vector<int>& layer_sizes);

		double relu(double x);

		std::vector<double> feedforward(const std::vector<double>& input);

		void setWeightsBiasesFromConstants();
	"""
    # We want the h Code to be pasted after
    #  /*! Initializes model */
    # void init();
    # So we need to find that string inside the .h-file
    search_string_h = "~"
    insert_line = None

    with open(h_path, 'r') as h_file, open(h_path_tmp, 'w') as h_file_tmp:
        lines = h_file.readlines()
        for i, line in enumerate(lines):
            if line.find(search_string_h) != -1:
                print(f"Found search string for .h-file: {search_string_h}")
                insert_line = lines.index(line)
                print(f"Line number: {lines.index(line)}")
            if insert_line is not None:
                if i == insert_line+1:
                    h_file_tmp.write(h_string)
            h_file_tmp.write(line)
    os.remove(h_path)
    os.rename(h_path_tmp, h_path)

    input_interface_variables = "\n".join([f"#define FMI_INPUT_{i} {i}"for i in range(1, n_inputs+1)])

    output_interface_variables = "\n" + "\n".join([f"#define FMI_OUTPUT_{i} {n_inputs+i}"for i in range(1, n_outputs+1)])

    input_init = "\n\t".join([f"m_realVar[FMI_INPUT_{i}] = 1.0;"for i in range(1, n_inputs+1)])
    output_init = "\n\t".join([f"m_realVar[FMI_OUTPUT_{i}] = 0.0;"for i in range(1, n_outputs+1)])

    input_string = "{" + ",\n\t\t".join([f"m_realVar[FMI_INPUT_{i}]"for i in range(1, n_inputs+1)]) + "}"
    output_string = "\n\t".join([f"m_realVar[FMI_OUTPUT_{i}] = prediction[{i-1}];"for i in range(1, n_outputs+1)])
    cpp_string_1 = """ /*

FMI Interface for FMU generated by FMICodeGenerator.

This file is part of FMICodeGenerator (https://github.com/ghorwin/FMICodeGenerator)

BSD 3-Clause License

Copyright (c) 2018, Andreas Nicolai
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#include <cstring>
#include <sstream>

#include "fmi2common/fmi2Functions.h"
#include "fmi2common/fmi2FunctionTypes.h"
#include "{MODELNAME}.h"
#include "weights_biases.h"

// FMI interface variables
{INPUT_INTERFACE_VARIABLES}
{OUTPUT_INTERFACE_VARIABLES}

// *** Variables and functions to be implemented in user code. ***

// *** GUID that uniquely identifies this FMU code
const char * const InstanceData::GUID = "{GUID}";

// *** Factory function, creates model specific instance of InstanceData-derived class
InstanceData * InstanceData::create() {{
    return new {MODELNAME}; // caller takes ownership
}}

void {MODELNAME}::neural_network(const std::vector<int>& layer_sizes) {{
    int num_layers = layer_sizes.size();
    weights.resize(num_layers - 1);
    biases.resize(num_layers - 1);
    activations.resize(num_layers);

    // Initialize weights and biases randomly (for simplicity)
    for (int i = 0; i < num_layers - 1; ++i) {{
        int num_neurons_in = layer_sizes[i];
        int num_neurons_out = layer_sizes[i + 1];

        // Initialize weights randomly (you can use a different initialization method)
        weights[i].resize(num_neurons_out, std::vector<double>(num_neurons_in));
        for (int j = 0; j < num_neurons_out; ++j) {{
            for (int k = 0; k < num_neurons_in; ++k) {{
                weights[i][j][k] = (rand() % 2000 - 1000) / 1000.0; // Random values between -1 and 1
            }}
        }}

        // Initialize biases to zeros
        biases[i].resize(num_neurons_out, 0.0);
    }}
}}

// ReLU activation function
double {MODELNAME}::relu(double x) {{
    return std::max(0.0, x);
}}

// Forward propagation for evaluation
std::vector<double> {MODELNAME}::feedforward(const std::vector<double>& input) {{
    int num_layers = activations.size();
    activations[0] = input;

    for (int i = 0; i < num_layers - 1; ++i) {{
        int num_neurons_in = weights[i].size();
        int num_neurons_out = weights[i][0].size();
        std::vector<double> layer_output(num_neurons_out, 0.0);

        for (int k = 0; k < num_neurons_out; ++k) {{
            double weighted_sum = biases[i][k];
            for (int j = 0; j < num_neurons_in; ++j) {{
                weighted_sum += weights[i][j][k] * activations[i][j];
            }}

            if (i < num_layers-2){{
                layer_output[k] = relu(weighted_sum);
            }} else {{
                layer_output[k] = weighted_sum;
            }}


        }}

        activations[i + 1] = layer_output;
    }}

    return activations.back();
}}

void {MODELNAME}::setWeightsBiasesFromConstants(){{
    weights = NeuralNetworkParams::weights;
    biases = NeuralNetworkParams::biases;
}}

{MODELNAME}::{MODELNAME}() :
    InstanceData()
{{
    // initialize input variables and/or parameters
    {INPUT_INIT}

    // initialize output variables
    {OUTPUT_INIT}

    neural_network(layer_sizes);
    setWeightsBiasesFromConstants();
}}


{MODELNAME}::~{MODELNAME}() {{
}}


// create a model instance
void {MODELNAME}::init() {{
    logger(fmi2OK, "progress", "Starting initialization.");

    if (m_modelExchange) {{
        // initialize states


        // TODO : implement your own initialization code here
    }}
    else {{
        // initialize states, these are used for our internal time integration


        // TODO : implement your own initialization code here

        // initialize integrator for co-simulation
        m_currentTimePoint = 0;
    }}

    logger(fmi2OK, "progress", "Initialization complete.");
}}


// model exchange: implementation of derivative and output update
void {MODELNAME}::updateIfModified() {{
    if (!m_externalInputVarsModified)
        return;

    // TODO : implement your code here
    std::vector<double> input = {INPUT_STRING};

    // Forward pass for evaluation
    std::vector<double> prediction = feedforward(input);

    // output variables
    {OUTPUT_STRING}


    // reset externalInputVarsModified flag
    m_externalInputVarsModified = false;
}}


// Co-simulation: time integration
void {MODELNAME}::integrateTo(double tCommunicationIntervalEnd) {{

    // state of FMU before integration:
    //   m_currentTimePoint = t_IntervalStart;

    // get input variables



    // TODO : implement your code here

    updateIfModified(); // re-use modelexchange code
    // output variables


    m_currentTimePoint = tCommunicationIntervalEnd;

    // state of FMU after integration:
    //   m_currentTimePoint = tCommunicationIntervalEnd;
}}


void {MODELNAME}::computeFMUStateSize() {{
    // store time, states and outputs
    m_fmuStateSize = sizeof(double)*1;
    // we store all cached variables

    // for all 4 maps, we store the size for sanity checks
    m_fmuStateSize += sizeof(int)*4;

    // serialization of the maps: first the valueRef, then the actual value

    m_fmuStateSize += (sizeof(int) + sizeof(double))*m_realVar.size();
    m_fmuStateSize += (sizeof(int) + sizeof(int))*m_integerVar.size();
    m_fmuStateSize += (sizeof(int) + sizeof(int))*m_boolVar.size(); // booleans are stored as int

    // strings are serialized in checkable format: first length, then zero-terminated string
    for (std::map<int, std::string>::const_iterator it = m_stringVar.begin();
        it != m_stringVar.end(); ++it)
    {{
        m_fmuStateSize += sizeof(int) + sizeof(int) + it->second.size() + 1; // add one char for \0
    }}


    // other variables: distinguish between ModelExchange and CoSimulation
    if (m_modelExchange) {{

        // TODO : store state variables and already computed derivatives

    }}
    else {{

        // TODO : store integrator state

    }}
}}


// macro for storing a POD and increasing the pointer to the linear memory array
#define SERIALIZE(type, storageDataPtr, value)\
{{\
*reinterpret_cast<type *>(storageDataPtr) = (value);\
(storageDataPtr) = reinterpret_cast<char *>(storageDataPtr) + sizeof(type);\
}}

// macro for retrieving a POD and increasing the pointer to the linear memory array
#define DESERIALIZE(type, storageDataPtr, value)\
{{\
(value) = *reinterpret_cast<type *>(storageDataPtr);\
(storageDataPtr) = reinterpret_cast<const char *>(storageDataPtr) + sizeof(type);\
}}


template <typename T>
bool deserializeMap({MODELNAME} * obj, const char * & dataPtr, const char * typeID, std::map<int, T> & varMap) {{
    // now de-serialize the maps: first the size (for checking), then each key-value pair
    int mapsize;
    DESERIALIZE(const int, dataPtr, mapsize);
    if (mapsize != static_cast<int>(varMap.size())) {{
        std::stringstream strm;
        strm << "Bad binary data or invalid/uninitialized model data. "<< typeID << "-Map size mismatch.";
        obj->logger(fmi2Error, "deserialization", strm.str());
        return false;
    }}
    for (int i=0; i<mapsize; ++i) {{
        int valueRef;
        T val;
        DESERIALIZE(const int, dataPtr, valueRef);
        if (varMap.find(valueRef) == varMap.end()) {{
            std::stringstream strm;
            strm << "Bad binary data or invalid/uninitialized model data. "<< typeID << "-Variable with value ref "<< valueRef
                << " does not exist in "<< typeID << "-variable map.";
            obj->logger(fmi2Error, "deserialization", strm.str());
            return false;
        }}
        DESERIALIZE(const T, dataPtr, val);
        varMap[valueRef] = val;
    }}
    return true;
}}



void {MODELNAME}::serializeFMUstate(void * FMUstate) {{
    char * dataPtr = reinterpret_cast<char*>(FMUstate);
    if (m_modelExchange) {{
        SERIALIZE(double, dataPtr, m_tInput);

        // TODO ModelExchange-specific serialization
    }}
    else {{
        SERIALIZE(double, dataPtr, m_currentTimePoint);

        // TODO CoSimulation-specific serialization
    }}

    // write map size for checking
    int mapSize = static_cast<int>(m_realVar.size());
    SERIALIZE(int, dataPtr, mapSize);
    // now serialize all members of the map
    for (std::map<int,double>::const_iterator it = m_realVar.begin(); it != m_realVar.end(); ++it) {{
        SERIALIZE(int, dataPtr, it->first);
        SERIALIZE(double, dataPtr, it->second);
    }}
    mapSize = static_cast<int>(m_integerVar.size());
    SERIALIZE(int, dataPtr, mapSize);
    for (std::map<int,int>::const_iterator it = m_integerVar.begin(); it != m_integerVar.end(); ++it) {{
        SERIALIZE(int, dataPtr, it->first);
        SERIALIZE(int, dataPtr, it->second);
    }}
    mapSize = static_cast<int>(m_boolVar.size());
    SERIALIZE(int, dataPtr, mapSize);
    for (std::map<int,int>::const_iterator it = m_boolVar.begin(); it != m_boolVar.end(); ++it) {{
        SERIALIZE(int, dataPtr, it->first);
        SERIALIZE(int, dataPtr, it->second);
    }}
    mapSize = static_cast<int>(m_stringVar.size());
    SERIALIZE(int, dataPtr, mapSize);
    for (std::map<int, std::string>::const_iterator it = m_stringVar.begin();
        it != m_stringVar.end(); ++it)
    {{
        SERIALIZE(int, dataPtr, it->first);				// map key
        SERIALIZE(int, dataPtr, static_cast<int>(it->second.size()));		// string size
        std::memcpy(dataPtr, it->second.c_str(), it->second.size()+1); // also copy the trailing \\0
        dataPtr += it->second.size()+1;
    }}
}}


bool {MODELNAME}::deserializeFMUstate(void * FMUstate) {{
    const char * dataPtr = reinterpret_cast<const char*>(FMUstate);
    if (m_modelExchange) {{
        DESERIALIZE(const double, dataPtr, m_tInput);

        // TODO ModelExchange-specific deserialization
        m_externalInputVarsModified = true;
    }}
    else {{
        DESERIALIZE(const double, dataPtr, m_currentTimePoint);

        // TODO CoSimulation-specific deserialization
    }}

    if (!deserializeMap(this, dataPtr, "real", m_realVar))
        return false;
    if (!deserializeMap(this, dataPtr, "integer", m_integerVar))
        return false;
    if (!deserializeMap(this, dataPtr, "boolean", m_boolVar))
        return false;

    // special handling for deserialization of string map
    int mapsize;
    DESERIALIZE(const int, dataPtr, mapsize);
    if (mapsize != static_cast<int>(m_stringVar.size())) {{
        logger(fmi2Error, "deserialization", "Bad binary data or invalid/uninitialized model data. string-variable map size mismatch.");
        return false;
    }}
    for (int i=0; i<mapsize; ++i) {{
        int valueRef;
        DESERIALIZE(const int, dataPtr, valueRef);
        if (m_stringVar.find(valueRef) == m_stringVar.end()) {{
            std::stringstream strm;
            strm << "Bad binary data or invalid/uninitialized model data. string-variable with value ref "<< valueRef
                << " does not exist in real variable map.";
            logger(fmi2Error, "deserialization", strm.str());
            return false;
        }}
        // get length of string
        int strLen;
        DESERIALIZE(const int, dataPtr, strLen);
        // create a string of requested length
        std::string s(static_cast<size_t>(strLen), ' ');
        // copy contents of string
        std::memcpy(&s[0], dataPtr, static_cast<size_t>(strLen)); // do not copy the trailing \\0
        dataPtr += strLen;
        // check that next character is a \\0
        if (*dataPtr != '\\0') {{
            std::stringstream strm;
            strm << "Bad binary data. string-variable with value ref "<< valueRef
                << " does not have a trailing \\0.";
            logger(fmi2Error, "deserialization", strm.str());
            return false;
        }}
        ++dataPtr;
        // replace value in map
        m_stringVar[valueRef] = s;
    }}

    return true;
}}
""".format(MODELNAME = modelName,
           GUID = guid,
           INPUT_INTERFACE_VARIABLES=input_interface_variables,
           OUTPUT_INTERFACE_VARIABLES=output_interface_variables,
           INPUT_INIT = input_init,
           OUTPUT_INIT = output_init,
           INPUT_STRING = input_string,
           OUTPUT_STRING = output_string)

    # print(cpp_string_1)

    os.remove(cpp_path)

    with open(cpp_path, 'w') as cpp_file:
        cpp_file.write(cpp_string_1)

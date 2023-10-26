 /*

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
#include "NN_Lorenz_CBO.h"
#include "weights_biases.h"

// FMI interface variables
#define FMI_INPUT_1 1
#define FMI_INPUT_2 2
#define FMI_INPUT_3 3

#define FMI_OUTPUT_1 4

// *** Variables and functions to be implemented in user code. ***

// *** GUID that uniquely identifies this FMU code
const char * const InstanceData::GUID = "{8899a7f8-6f51-11ee-89f9-00155d6dc665}";

// *** Factory function, creates model specific instance of InstanceData-derived class
InstanceData * InstanceData::create() {
    return new NN_Lorenz_CBO; // caller takes ownership
}

void NN_Lorenz_CBO::neural_network(const std::vector<int>& layer_sizes) {
    int num_layers = layer_sizes.size();
    weights.resize(num_layers - 1);
    biases.resize(num_layers - 1);
    activations.resize(num_layers);

    // Initialize weights and biases randomly (for simplicity)
    for (int i = 0; i < num_layers - 1; ++i) {
        int num_neurons_in = layer_sizes[i];
        int num_neurons_out = layer_sizes[i + 1];

        // Initialize weights randomly (you can use a different initialization method)
        weights[i].resize(num_neurons_out, std::vector<double>(num_neurons_in));
        for (int j = 0; j < num_neurons_out; ++j) {
            for (int k = 0; k < num_neurons_in; ++k) {
                weights[i][j][k] = (rand() % 2000 - 1000) / 1000.0; // Random values between -1 and 1
            }
        }

        // Initialize biases to zeros
        biases[i].resize(num_neurons_out, 0.0);
    }
}

// ReLU activation function
double NN_Lorenz_CBO::relu(double x) {
    return std::max(0.0, x);
}

// Forward propagation for evaluation
std::vector<double> NN_Lorenz_CBO::feedforward(const std::vector<double>& input) {
    int num_layers = activations.size();
    activations[0] = input;

    for (int i = 0; i < num_layers - 1; ++i) {
        int num_neurons_in = weights[i].size();
        int num_neurons_out = weights[i][0].size();
        std::vector<double> layer_output(num_neurons_out, 0.0);

        for (int k = 0; k < num_neurons_out; ++k) {
            double weighted_sum = biases[i][k];
            for (int j = 0; j < num_neurons_in; ++j) {
                weighted_sum += weights[i][j][k] * activations[i][j];
            }

            if (i < num_layers-2){
                layer_output[k] = relu(weighted_sum);
            } else {
                layer_output[k] = weighted_sum;
            }


        }

        activations[i + 1] = layer_output;
    }

    return activations.back();
}

void NN_Lorenz_CBO::setWeightsBiasesFromConstants(){
    weights = NeuralNetworkParams::weights;
    biases = NeuralNetworkParams::biases;
}

NN_Lorenz_CBO::NN_Lorenz_CBO() :
    InstanceData()
{
    // initialize input variables and/or parameters
    m_realVar[FMI_INPUT_1] = 1.0;
	m_realVar[FMI_INPUT_2] = 1.0;
	m_realVar[FMI_INPUT_3] = 1.0;

    // initialize output variables
    m_realVar[FMI_OUTPUT_1] = 0.0;

    neural_network(layer_sizes);
    setWeightsBiasesFromConstants();
}


NN_Lorenz_CBO::~NN_Lorenz_CBO() {
}


// create a model instance
void NN_Lorenz_CBO::init() {
    logger(fmi2OK, "progress", "Starting initialization.");

    if (m_modelExchange) {
        // initialize states


        // TODO : implement your own initialization code here
    }
    else {
        // initialize states, these are used for our internal time integration


        // TODO : implement your own initialization code here

        // initialize integrator for co-simulation
        m_currentTimePoint = 0;
    }

    logger(fmi2OK, "progress", "Initialization complete.");
}


// model exchange: implementation of derivative and output update
void NN_Lorenz_CBO::updateIfModified() {
    if (!m_externalInputVarsModified)
        return;

    // TODO : implement your code here
    std::vector<double> input = {m_realVar[FMI_INPUT_1],
		m_realVar[FMI_INPUT_2],
		m_realVar[FMI_INPUT_3]};

    // Forward pass for evaluation
    std::vector<double> prediction = feedforward(input);

    // output variables
    m_realVar[FMI_OUTPUT_1] = prediction[0];


    // reset externalInputVarsModified flag
    m_externalInputVarsModified = false;
}


// Co-simulation: time integration
void NN_Lorenz_CBO::integrateTo(double tCommunicationIntervalEnd) {

    // state of FMU before integration:
    //   m_currentTimePoint = t_IntervalStart;

    // get input variables



    // TODO : implement your code here

    updateIfModified(); // re-use modelexchange code
    // output variables


    m_currentTimePoint = tCommunicationIntervalEnd;

    // state of FMU after integration:
    //   m_currentTimePoint = tCommunicationIntervalEnd;
}


void NN_Lorenz_CBO::computeFMUStateSize() {
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
    {
        m_fmuStateSize += sizeof(int) + sizeof(int) + it->second.size() + 1; // add one char for  
    }


    // other variables: distinguish between ModelExchange and CoSimulation
    if (m_modelExchange) {

        // TODO : store state variables and already computed derivatives

    }
    else {

        // TODO : store integrator state

    }
}


// macro for storing a POD and increasing the pointer to the linear memory array
#define SERIALIZE(type, storageDataPtr, value){*reinterpret_cast<type *>(storageDataPtr) = (value);(storageDataPtr) = reinterpret_cast<char *>(storageDataPtr) + sizeof(type);}

// macro for retrieving a POD and increasing the pointer to the linear memory array
#define DESERIALIZE(type, storageDataPtr, value){(value) = *reinterpret_cast<type *>(storageDataPtr);(storageDataPtr) = reinterpret_cast<const char *>(storageDataPtr) + sizeof(type);}


template <typename T>
bool deserializeMap(NN_Lorenz_CBO * obj, const char * & dataPtr, const char * typeID, std::map<int, T> & varMap) {
    // now de-serialize the maps: first the size (for checking), then each key-value pair
    int mapsize;
    DESERIALIZE(const int, dataPtr, mapsize);
    if (mapsize != static_cast<int>(varMap.size())) {
        std::stringstream strm;
        strm << "Bad binary data or invalid/uninitialized model data. "<< typeID << "-Map size mismatch.";
        obj->logger(fmi2Error, "deserialization", strm.str());
        return false;
    }
    for (int i=0; i<mapsize; ++i) {
        int valueRef;
        T val;
        DESERIALIZE(const int, dataPtr, valueRef);
        if (varMap.find(valueRef) == varMap.end()) {
            std::stringstream strm;
            strm << "Bad binary data or invalid/uninitialized model data. "<< typeID << "-Variable with value ref "<< valueRef
                << " does not exist in "<< typeID << "-variable map.";
            obj->logger(fmi2Error, "deserialization", strm.str());
            return false;
        }
        DESERIALIZE(const T, dataPtr, val);
        varMap[valueRef] = val;
    }
    return true;
}



void NN_Lorenz_CBO::serializeFMUstate(void * FMUstate) {
    char * dataPtr = reinterpret_cast<char*>(FMUstate);
    if (m_modelExchange) {
        SERIALIZE(double, dataPtr, m_tInput);

        // TODO ModelExchange-specific serialization
    }
    else {
        SERIALIZE(double, dataPtr, m_currentTimePoint);

        // TODO CoSimulation-specific serialization
    }

    // write map size for checking
    int mapSize = static_cast<int>(m_realVar.size());
    SERIALIZE(int, dataPtr, mapSize);
    // now serialize all members of the map
    for (std::map<int,double>::const_iterator it = m_realVar.begin(); it != m_realVar.end(); ++it) {
        SERIALIZE(int, dataPtr, it->first);
        SERIALIZE(double, dataPtr, it->second);
    }
    mapSize = static_cast<int>(m_integerVar.size());
    SERIALIZE(int, dataPtr, mapSize);
    for (std::map<int,int>::const_iterator it = m_integerVar.begin(); it != m_integerVar.end(); ++it) {
        SERIALIZE(int, dataPtr, it->first);
        SERIALIZE(int, dataPtr, it->second);
    }
    mapSize = static_cast<int>(m_boolVar.size());
    SERIALIZE(int, dataPtr, mapSize);
    for (std::map<int,int>::const_iterator it = m_boolVar.begin(); it != m_boolVar.end(); ++it) {
        SERIALIZE(int, dataPtr, it->first);
        SERIALIZE(int, dataPtr, it->second);
    }
    mapSize = static_cast<int>(m_stringVar.size());
    SERIALIZE(int, dataPtr, mapSize);
    for (std::map<int, std::string>::const_iterator it = m_stringVar.begin();
        it != m_stringVar.end(); ++it)
    {
        SERIALIZE(int, dataPtr, it->first);				// map key
        SERIALIZE(int, dataPtr, static_cast<int>(it->second.size()));		// string size
        std::memcpy(dataPtr, it->second.c_str(), it->second.size()+1); // also copy the trailing \0
        dataPtr += it->second.size()+1;
    }
}


bool NN_Lorenz_CBO::deserializeFMUstate(void * FMUstate) {
    const char * dataPtr = reinterpret_cast<const char*>(FMUstate);
    if (m_modelExchange) {
        DESERIALIZE(const double, dataPtr, m_tInput);

        // TODO ModelExchange-specific deserialization
        m_externalInputVarsModified = true;
    }
    else {
        DESERIALIZE(const double, dataPtr, m_currentTimePoint);

        // TODO CoSimulation-specific deserialization
    }

    if (!deserializeMap(this, dataPtr, "real", m_realVar))
        return false;
    if (!deserializeMap(this, dataPtr, "integer", m_integerVar))
        return false;
    if (!deserializeMap(this, dataPtr, "boolean", m_boolVar))
        return false;

    // special handling for deserialization of string map
    int mapsize;
    DESERIALIZE(const int, dataPtr, mapsize);
    if (mapsize != static_cast<int>(m_stringVar.size())) {
        logger(fmi2Error, "deserialization", "Bad binary data or invalid/uninitialized model data. string-variable map size mismatch.");
        return false;
    }
    for (int i=0; i<mapsize; ++i) {
        int valueRef;
        DESERIALIZE(const int, dataPtr, valueRef);
        if (m_stringVar.find(valueRef) == m_stringVar.end()) {
            std::stringstream strm;
            strm << "Bad binary data or invalid/uninitialized model data. string-variable with value ref "<< valueRef
                << " does not exist in real variable map.";
            logger(fmi2Error, "deserialization", strm.str());
            return false;
        }
        // get length of string
        int strLen;
        DESERIALIZE(const int, dataPtr, strLen);
        // create a string of requested length
        std::string s(static_cast<size_t>(strLen), ' ');
        // copy contents of string
        std::memcpy(&s[0], dataPtr, static_cast<size_t>(strLen)); // do not copy the trailing \0
        dataPtr += strLen;
        // check that next character is a \0
        if (*dataPtr != '\0') {
            std::stringstream strm;
            strm << "Bad binary data. string-variable with value ref "<< valueRef
                << " does not have a trailing \0.";
            logger(fmi2Error, "deserialization", strm.str());
            return false;
        }
        ++dataPtr;
        // replace value in map
        m_stringVar[valueRef] = s;
    }

    return true;
}

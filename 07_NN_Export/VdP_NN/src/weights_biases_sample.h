// weights_biases.h

#ifndef WEIGHTS_BIASES_H
#define WEIGHTS_BIASES_H

#include <vector>

namespace NeuralNetworkParams {
    const std::vector<std::vector<std::vector<double>>> weights = {
        // Layer 0 weights
        {
            {0.1, 0.2, 0.3},
            {0.4, 0.5, 0.6}
        },
        // Layer 1 weights
        {
            {0.9},
            {0.1}
        }
    };

    const std::vector<std::vector<double>> biases = {
        // Layer 0 biases
        {0.7, 0.8},
        // Layer 1 biases
        {0.2}
    };
}

#endif

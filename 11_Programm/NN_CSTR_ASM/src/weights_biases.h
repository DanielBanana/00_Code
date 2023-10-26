#ifndef WEIGHTS_BIASES_H
#define WEIGHTS_BIASES_H

#include <vector>

namespace NeuralNetworkParams {
	const std::vector<std::vector<std::vector<double>>> weights = {
		// layers_0 weights
		{
			{0.0, 0.0, -0.0, 0.0, -0.0, -0.0, 0.0, 0.0, -0.0, -0.0, 0.0, -0.0, -0.0, -0.0, -0.0},
			{0.0, -0.0, 0.0, -0.0, 0.0, -0.0, -0.0, 0.0, -0.0, -0.0, 0.0, -0.0, 0.0, 0.0, 0.0},
		},
		// layers_1 weights
		{
			{-0.0},
			{-0.0},
			{-0.0},
			{-0.0},
			{0.0},
			{0.0},
			{-0.0},
			{0.0},
			{0.0},
			{0.0},
			{-0.0},
			{0.0},
			{-0.0},
			{-0.0},
			{-0.0},
		},
	};

	const std::vector<std::vector<double>> biases = {
		// layers_0 biases
		{0.0, -0.0, 0.0, -0.0, 0.0, 0.0, -0.0, 0.0, -0.0, -0.0, -0.0, 0.0, 0.0, 0.0, 0.0},
		// layers_1 biases
		{0.30487628668941186},
	};
}
std::vector<int> layer_sizes = {2, 15, 1};

#endif

#ifndef WEIGHTS_BIASES_H
#define WEIGHTS_BIASES_H

#include <vector>

namespace NeuralNetworkParams {
	const std::vector<std::vector<std::vector<double>>> weights = {
		// layers_0 weights
		{
			{nan, nan, nan},
			{nan, nan, nan},
			{nan, nan, nan},
			{nan, nan, nan},
			{nan, nan, nan},
			{nan, nan, nan},
			{nan, nan, nan},
			{nan, nan, nan},
			{nan, nan, nan},
			{nan, nan, nan},
			{nan, nan, nan},
			{nan, nan, nan},
			{nan, nan, nan},
			{nan, nan, nan},
			{nan, nan, nan},
		},
		// layers_2 weights
		{
			{nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan},
		},
	};

	const std::vector<std::vector<double>> biases = {
		// layers_0 biases
		{nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan},
		// layers_2 biases
		{nan},
	};
}
std::vector<int> layer_sizes = {15, 1, 15};

#endif

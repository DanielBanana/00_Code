double weights1[15][2] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
double bias1[15] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
double weights2[1][15] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
double bias2[1] = {1};

// Define the neural network architecture (e.g., 2 input, 4 hidden, 1 output)
std::vector<int> layer_sizes = {2, 15, 1};
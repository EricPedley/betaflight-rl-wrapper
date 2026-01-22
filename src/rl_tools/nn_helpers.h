#include <math.h>

void nn_linear(const float* weights, const float* biases, const float* input, int in_features, int out_features, float* output);

void nn_elu(float* input, int size);

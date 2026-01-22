#include "nn_helpers.h"

void nn_linear(const float* weights, const float* biases, const float* input, int in_features, int out_features, float* output) {
    for (int i = 0; i < out_features; ++i) {
        float neuron = biases[i];
        for (int j = 0; j < in_features; ++j) {
            neuron += input[j] * weights[i * in_features + j];
        }
        output[i] = neuron;
    }
}

void nn_elu(float* input, int size) {
    for (int i = 0; i < size; ++i) {
        if (input[i] < 0) {
            input[i] = expf(input[i]) - 1;
        } else {
            input[i] = input[i];
        }
    }
}


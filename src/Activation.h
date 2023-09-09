#pragma once


#include <cmath>


enum Activation {
    RELU,
    LEAKY_RELU,
    SIGMOID,
    TANH, // NOTE has been producing nans, probably wrong formula
    NO_ACTIVATION
};

float relu_activation(const float value) {
    return std::max(value, 0.0f);
}
float leaky_relu_activation(const float value) {
    if (value < 0) {
        return value * -0.1f;
    }
    return value;
}
float sigmoid_activation(const float value) {
    return 1.0f / (1.0f + std::exp(-value));
}
// NOTE has been producing nans, probably wrong formula
float tanh_activation(const float value) {
    return (std::exp(value) - std::exp(-value)) / (std::exp(value) + std::exp(-value));
}
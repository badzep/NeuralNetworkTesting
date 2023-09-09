#pragma once


#include <cmath>
#include <iostream>
#include <array>


#include "Activation.h"


template<const Activation INPUT_ACTIVATION, const Activation HIDDEN_ACTIVATION, const Activation OUTPUT_ACTIVATION, const unsigned short ...LAYER_SIZES>
class Network {
private:
    constexpr static unsigned short layer_count() {
        unsigned short count = 0;
        for (unsigned short v: {LAYER_SIZES...}) {
            count++;
        }
        return count;
    }

    constexpr static std::array<unsigned short, layer_count()> layer_sizes() {
        std::array<unsigned short, layer_count()> x = {LAYER_SIZES...};
        return x;
    }

    constexpr static unsigned short layer_size_at(const unsigned short index) {
        return layer_sizes()[index];
    }

    constexpr static unsigned short neuron_count() {
        unsigned short count = 0;
        for (unsigned short layer_index = 0; layer_index < layer_count(); layer_index++) {
            count += layer_size_at(layer_index);
        }
        return count;
    }

    constexpr static unsigned short weight_count() {
        unsigned short count = 0;
        for (unsigned short layer_index = 1; layer_index < layer_count(); layer_index++) {
            count += layer_size_at(layer_index) * layer_size_at(layer_index - 1);
        }
        return count;
    }

    constexpr Activation get_layer_activation(const unsigned short layer_index) {
        if (layer_index == layer_count() - 1) {
            return OUTPUT_ACTIVATION;
        }
        if (layer_index == 0) {
            return INPUT_ACTIVATION;
        }
        return HIDDEN_ACTIVATION;
    }

    constexpr static unsigned short layer_start_index(const unsigned short target_layer_index) {
        unsigned short count = 0;
        for (unsigned short layer_index = 0; layer_index < target_layer_index; layer_index++) {
            count += layer_size_at(layer_index);
        }
        return count;
    }

    float values[neuron_count()];
    float weights[weight_count()];
    float biases[neuron_count()];

    [[nodiscard]] float get_value_at(const unsigned short target_layer_index, const unsigned short neuron_index) const {
        return this->values[layer_start_index(target_layer_index) + neuron_index];
    }

    // NOTE: you can't get weights at layer 0 (because they don't exist)
    [[nodiscard]] float get_weight_at(const unsigned short target_layer_index, const unsigned short neuron_index, const unsigned short input_index) {
        unsigned short count = 0;
        for (unsigned short layer_index = 1; layer_index < target_layer_index; layer_index++) {
            count += layer_size_at(layer_index) * layer_size_at(layer_index - 1);
        }
        count += neuron_index * layer_size_at(target_layer_index - 1);
        count += input_index;
        return this->weights[count];
    }

    void apply_activation(const unsigned short layer_index) {
        switch (this->get_layer_activation(layer_index)) {
            case RELU:
                for (unsigned short neuron_index = 0; neuron_index < layer_size_at(layer_index); neuron_index++) {
                    this->values[layer_start_index(layer_index) + neuron_index] = relu_activation(this->values[layer_start_index(layer_index) + neuron_index]);
                }
                return;
            case LEAKY_RELU:
                for (unsigned short neuron_index = 0; neuron_index < layer_size_at(layer_index); neuron_index++) {
                    this->values[layer_start_index(layer_index) + neuron_index] = leaky_relu_activation(this->values[layer_start_index(layer_index) + neuron_index]);
                }
                return;
            case SIGMOID:
                for (unsigned short neuron_index = 0; neuron_index < layer_size_at(layer_index); neuron_index++) {
                    this->values[layer_start_index(layer_index) + neuron_index] = sigmoid_activation(this->values[layer_start_index(layer_index) + neuron_index]);
                }
                return;
            case TANH:
                for (unsigned short neuron_index = 0; neuron_index < layer_size_at(layer_index); neuron_index++) {
                    this->values[layer_start_index(layer_index) + neuron_index] = tanh_activation(this->values[layer_start_index(layer_index) + neuron_index]);
                }
                return;
            case NO_ACTIVATION:
                return;
        }
    }

    void pass_layer(const unsigned short layer_index) {
        for (unsigned short neuron_index = 0; neuron_index < layer_size_at(layer_index); neuron_index++) {
            for (unsigned short input_neuron_index = 0; input_neuron_index < layer_size_at(layer_index - 1); input_neuron_index++) {
                this->values[layer_start_index(layer_index) + neuron_index] += this->values[layer_start_index(layer_index - 1) + input_neuron_index] * this->get_weight_at(layer_index, neuron_index, input_neuron_index);
            }
        }
        this->apply_activation(layer_index);
    }

public:
    Network() = default;

    Network(const float _weights[weight_count()], const float _biases[neuron_count()]) {
        for (unsigned short neuron_index = 0; neuron_index < neuron_count(); neuron_index++) {
            this->values[neuron_index] = 0;
        }
        for (unsigned short neuron_index = 0; neuron_index < weight_count(); neuron_index++) {
            this->weights[neuron_index] = _weights[neuron_index];
        }
        for (unsigned short neuron_index = 0; neuron_index < neuron_count(); neuron_index++) {
            this->biases[neuron_index] = _biases[neuron_index];
        }
    }

    ~Network() = default;

    void reset() {
        for (unsigned short neuron_index = 0; neuron_index < neuron_count(); neuron_index++) {
            this->values[neuron_index] = 0;
        }
    }

    void pass() {
        for (unsigned short neuron_index = 0; neuron_index < neuron_count(); neuron_index++) {
            this->values[neuron_index] += this->biases[neuron_index];
        }
        for (unsigned short layer_index = 1; layer_index < layer_count(); layer_index++) {
            this->pass_layer(layer_index);
        }
    }

    void set_input(const unsigned short input_index, const float value) {
        if (input_index >= layer_size_at(0)) {
            printf("-----------------BAD INPUT--------------------\n");
        }
        this->values[input_index] = value;
    }

    [[nodiscard]] float get_output(const unsigned short output_index) const {
        return get_value_at(layer_count() - 1, output_index);
    }

    void print_values() const {
        printf("Values: ");
        for (unsigned short neuron_index = 0; neuron_index < neuron_count(); neuron_index++) {
            printf("%f ", this->values[neuron_index]);
        }
        printf("\n");
    }

    void print_parameters() const {
        printf("Weights: ");
        for (unsigned short weight_index = 0; weight_index < weight_count(); weight_index++) {
            printf("%f ", this->weights[weight_index]);
        }
        printf("\n");
        printf("Biases: ");
        for (unsigned short neuron_index = 0; neuron_index < neuron_count(); neuron_index++) {
            printf("%f ", this->biases[neuron_index]);
        }
        printf("\n");
    }

    void print_output() const {
        printf("Outputs: ");
        for (unsigned short neuron_index = layer_start_index(layer_count() - 1); neuron_index < neuron_count(); neuron_index++) {
            printf("%f ", this->values[neuron_index]);
        }
        printf("\n");
    }
};
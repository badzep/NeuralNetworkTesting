#pragma once


#include <cmath>
#include <vector>
#include <random>


#include "RNG.h"
#include "Activation.h"


constexpr float MUTATION_MULTIPLIER = 1;

constexpr unsigned short INPUT_COUNT = 5;
constexpr unsigned short OUTPUT_COUNT = 6;
constexpr unsigned short RESERVED_COUNT = INPUT_COUNT + OUTPUT_COUNT;

constexpr float NEW_RANDOM_LINK_CHANCE = 25.0f; // out of 100
constexpr float REMOVE_RANDOM_LINK_CHANCE = 10.0f; // out of 100
constexpr float NEW_NEURON_CHANCE = 5; // out of 100

constexpr float MAX_ACTIVATION_VALUE = 1e3;

std::normal_distribution<float> weight_mutation_distribution_mult(1, .05f * MUTATION_MULTIPLIER);
std::normal_distribution<float> weight_mutation_distribution_add(.08, .08f * MUTATION_MULTIPLIER);

std::normal_distribution<float> bias_mutation_distribution_mult(1, .05f * MUTATION_MULTIPLIER);
std::normal_distribution<float> bias_mutation_distribution_add(0.01, 0.01f * MUTATION_MULTIPLIER);

std::normal_distribution<float> retention_mutation_distribution_mult(1, .07f * MUTATION_MULTIPLIER);
std::normal_distribution<float> retention_mutation_distribution_add(.01, .001f * MUTATION_MULTIPLIER);


std::normal_distribution<float> weight_distribution(0.3f, 0.2f);
std::normal_distribution<float> retention_distribution(0.1f, 0.01f);
std::normal_distribution<float> bias_distribution(0.1f, 0.075f);


std::uniform_int_distribution<unsigned short> initial_neuron_count_distribution(3, 10);
std::uniform_int_distribution<unsigned short> initial_random_link_count_distribution(0, 0);

std::uniform_real_distribution<float> uniform_percent(0, 100);


// NOTE links can go in both directions possibly creating loops
class Link {
public:
    unsigned short start_neuron_index;
    unsigned short end_neuron_index;
    float weight;
};

// Retention indicates how much of the neuron's value remains between passes. 0 = None, 1 = All
class RetentiveNeuron {
public:
    float value;
    float retention;
    float bias;

    void prepare() {
        this->value *= retention;
        this->value += bias;
    }
};

// Neuron values are not cleared between passes, activation functions are only applied to output neurons, and links can go from any neuron to any neuron
// This structure is meant to more closely resemble an actual brain and theoretically allows information to be stored inside the network between passes
class RetentiveNetwork {
public:
    std::vector<RetentiveNeuron> neurons;
    std::vector<Link> links;

    RetentiveNetwork() = default;

    ~RetentiveNetwork() = default;

    void copy_structure(std::vector<RetentiveNeuron> &parent_neurons, std::vector<Link> &parent_links) {
        for (RetentiveNeuron &neuron: parent_neurons) {
            this->neurons.push_back({neuron.value, neuron.retention, neuron.bias});
        }
        for (Link &link: parent_links) {
            this->links.push_back({link.start_neuron_index, link.end_neuron_index, link.weight});
        }
    }

    void add_reserved_neurons() {
        for (unsigned short index = 0; index < INPUT_COUNT; index++) {
            this->neurons.push_back({0.0f, 0.0f, 0.0f});
        }
        for (unsigned short index = 0; index < OUTPUT_COUNT; index++) {
            this->neurons.push_back({0.0f, 0.0f, 0.0f});
        }
    }

    void add_hidden_neuron() {
        this->neurons.push_back({0.0f, retention_distribution(RNG), bias_distribution(RNG)});
    }

    void add_hidden_neurons(const unsigned short hidden_neuron_count) {
        for (unsigned short index = 0; index < hidden_neuron_count; index++) {
            this->add_hidden_neuron();
        }
    }

    void add_random_link() {
        std::uniform_int_distribution<> neuron_distribution(0, (int) this->neurons.size() - 1);
        const unsigned short start = neuron_distribution(RNG);
        std::uniform_int_distribution<> neuron_distribution2(1, (int) this->neurons.size() - 1);
        unsigned short end = neuron_distribution(RNG);
        if (start == end) {
            end = 0;
        }
        this->links.push_back({start, end, weight_distribution(RNG)});
    }

    void add_random_links(const unsigned short new_link_count) {
        for (unsigned short index = 0; index < new_link_count; index++) {
            this->add_random_link();
        }
    }

    void remove_random_link() {
        std::uniform_int_distribution<> link_distribution(0, (int) this->links.size());
        this->links.erase(this->links.begin() + link_distribution(RNG));
    }

    void mutate() {
        if (uniform_percent(RNG) <= NEW_NEURON_CHANCE) {
            this->add_hidden_neuron();
        }

        if (uniform_percent(RNG) <= NEW_RANDOM_LINK_CHANCE) {
            this->add_random_link();
        }

        if (uniform_percent(RNG) <= REMOVE_RANDOM_LINK_CHANCE) {
            this->remove_random_link();
        }

        for (Link &link: this->links) {
            link.weight *= weight_mutation_distribution_mult(RNG);
            link.weight += weight_mutation_distribution_add(RNG);
        }

        for (RetentiveNeuron &neuron: this->neurons) {
            neuron.bias *= bias_mutation_distribution_mult(RNG);
            neuron.bias += bias_mutation_distribution_add(RNG);

            neuron.retention *= retention_mutation_distribution_mult(RNG);
            neuron.retention += retention_mutation_distribution_add(RNG);
        }
    }



    void initialize() {
        this->add_reserved_neurons();
        this->add_hidden_neurons(initial_neuron_count_distribution(RNG));
        this->add_random_links(initial_random_link_count_distribution(RNG));
    }

    float get_output_at(unsigned short output_index) {
        return this->neurons[INPUT_COUNT + output_index].value;
    }

    void apply_output_activation() {
        for (unsigned short i = INPUT_COUNT; i < INPUT_COUNT + OUTPUT_COUNT; i++) {
            this->neurons[i].value = sigmoid_activation(this->neurons[i].value);
        }
    }

    // Should be called before adding input values each for each pass
    void prepare() {
        for (RetentiveNeuron &neuron : this->neurons) {
            neuron.prepare();
        }
    }

    // All links pass only one time, this means that it may take multiple calls for a signal to propagate from an input neuron to an output neuron
    void pass() {
        for (Link &link: this->links) {
            this->neurons[link.end_neuron_index].value += this->neurons[link.start_neuron_index].value * link.weight;
            this->neurons[link.end_neuron_index].value = std::min(this->neurons[link.end_neuron_index].value, MAX_ACTIVATION_VALUE);
            this->neurons[link.end_neuron_index].value = std::max(this->neurons[link.end_neuron_index].value, -MAX_ACTIVATION_VALUE);
        }
        this->apply_output_activation();
    }
};
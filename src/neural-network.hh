#pragma once

#include <functional>
#include <vector>
#include "neuron.hh"

template <typename T>
class NeuralNetwork
{
public:
    using layer_type = std::vector<Neuron<T>>;

    NeuralNetwork(unsigned int input_nb, unsigned int hidden_nb,
                  unsigned int output_nb, std::function<T(T)> activate,
                  std::function<T(T)> activate_prime);

    void feed_forward(std::vector<T> inputs);
    T compute(std::vector<T> inputs);
private:
    layer_type input_layer_;
    layer_type hidden_layer_;
    layer_type output_layer_;

    const std::function<T(T)> activate_;
    const std::function<T(T)> activate_prime_;
};

#include "neural-network.hxx"

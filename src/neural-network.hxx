#pragma once

#include <functional>
#include <vector>
#include <random>
#include <exception>
#include <algorithm>
#include "neuron.hh"

#include <iostream>

template <typename T>
NeuralNetwork<T>::NeuralNetwork(unsigned int input_nb, unsigned int hidden_nb,
                  unsigned int output_nb, std::function<T(T)> activate,
                  std::function<T(T)> activate_prime)
    : input_layer_(input_nb + 1, Neuron<T>(hidden_nb)) // + 1 for the bias
    , hidden_layer_(hidden_nb + 1, Neuron<T>(output_nb)) // + 1 for the bias
    , output_layer_(output_nb, Neuron<T>(1))
    , activate_(activate)
    , activate_prime_(activate_prime)
{
    std::random_device rd;
    std::mt19937 gen(rd());

    std::normal_distribution<T> distrib;
    for (auto& n : input_layer_)
        for (auto& w : n.out_weights_)
            w = distrib(gen);

    for (auto& n : hidden_layer_)
        for (auto& w : n.out_weights_)
            w = distrib(gen);

    for (auto& n : output_layer_)
        for (auto& w : n.out_weights_)
            w = distrib(gen);
}

template <typename T>
void NeuralNetwork<T>::feed_forward(std::vector<T> inputs)
{
    if (inputs.size() + 1 != input_layer_.size())
        throw std::invalid_argument("Incorrect inputs amount.");

    for (typename std::vector<T>::size_type i = 0; i < inputs.size(); ++i)
        input_layer_[i].feed(inputs[i]);
    input_layer_.back().feed(1); // bias

    for (typename std::vector<T>::size_type i = 0; i < hidden_layer_.size() - 1; ++i)
    {
        T sum = 0;
        for (auto& n : input_layer_)
            sum += activate_(n.outputs_[i]);
        hidden_layer_[i].feed(sum);
    }
    hidden_layer_.back().feed(1); // bias

    for (typename std::vector<T>::size_type i = 0; i < output_layer_.size(); ++i)
    {
        T sum = 0;
        for (auto& n : hidden_layer_)
            sum += activate_(n.outputs_[i]);
        output_layer_[i].feed(sum);
    }
}

template <typename T>
T NeuralNetwork<T>::compute(std::vector<T> inputs)
{
    feed_forward(inputs);
    return output_layer_[0].outputs_[0];
}

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
    : input_layer_(input_nb + 1, Neuron<T>(hidden_nb, activate)) // + 1 for the bias
    , hidden_layer_(hidden_nb + 1, Neuron<T>(output_nb, activate)) // + 1 for the bias
    , output_layer_(output_nb, Neuron<T>(1, activate))
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
            sum += n.activated_outputs_[i];

        hidden_layer_[i].feed(sum);
    }
    hidden_layer_.back().feed(1); // bias

    for (typename std::vector<T>::size_type i = 0; i < output_layer_.size(); ++i)
    {
        T sum = 0;
        for (auto& n : hidden_layer_)
        {
            n.activated_outputs_[i] = activate_(n.outputs_[i]);
            sum += n.activated_outputs_[i];
        }

        output_layer_[i].feed(sum);
    }
}

template <typename T>
void NeuralNetwork<T>::back_propagate(T error)
{
    double learning_rate = 0.17;

//  OutputType delta_output_layer = activate_prime_(res.output_sum_) * error;
//  // compute changes in neurons output weight
//      InputsType hidden_output_changes;
//  std::transform(res.hidden_results_.begin(), res.hidden_results_.end(),
//                 std::back_inserter(hidden_output_changes),
//                 [&delta_output_layer, &learning_rate](const auto& hidden_res)
//                 { return (delta_output_layer / hidden_res) * learning_rate; });
//  double learning_rate = 0.17;

//  // compute delta hidden changes
//      InputsType delta_hidden_layers;
//  for (unsigned int i = 0; i < neurons_amount_; ++i)
//          delta_hidden_layers.push_back((delta_output_layer
//                                       / neurons_[i].output_weight_)
//                                       * activate_prime_(res.hidden_sums_[i]));


//  // update input weights
//  for (unsigned int i = 0; i < input_amount_; ++i)
//      for (unsigned int j = 0; j < neurons_amount_; ++j)
//              neurons_[j].input_weight_[i] += delta_hidden_layers[j] / inputs[i];
// -
//  // update neurons output weight
//  for (unsigned int i = 0; i < neurons_amount_; ++i)
//      neurons_[i].output_weight_ += hidden_output_changes[i];
// -
//  return error;

    for (typename std::vector<T>::size_type i = 0; i < output_layer_.size(); ++i)
    {
        for (auto& n : hidden_layer_)
        {
            double delta_out = activate_prime_(n.activated_outputs_[i]) * error;
            double new_weight = n.out_weights_[i] + learning_rate * delta_out / (n.outputs_[i] / n.out_weights_[i]);
            n.out_weights_[i] = new_weight;
//            std::cout << "old: " << n.out_weights_[i] << " new: " << new_weight << std::endl;
        }
    }
}

template <typename T>
void NeuralNetwork<T>::train(std::vector<T> inputs, T target)
{
    back_propagate(compute(inputs) - target);
}

template <typename T>
T NeuralNetwork<T>::compute(std::vector<T> inputs)
{
    feed_forward(inputs);
    return output_layer_[0].outputs_[0];
}

#pragma once

#include <functional>
#include <vector>
#include <random>
#include <exception>
#include <algorithm>

#include <iostream>

template <typename InputType, typename OutputType>
NeuralNetwork<InputType, OutputType>::NeuralNetwork(
    unsigned int input_amount, unsigned int output_amount,
    unsigned int neurons_amount,
    const std::function<OutputType(InputType)>& activation_func,
    const std::function<OutputType(InputType)>& prime_activation_func)
    : input_amount_(input_amount)
    , output_amount_(output_amount)
    , neurons_amount_(neurons_amount)
    , neurons_(neurons_amount)
    , activate_(activation_func)
    , activate_prime_(prime_activation_func)
{
    // Initialize neurons with random weights
    for (auto& neuron : neurons_)
    {
        std::random_device rd;
        std::mt19937 gen(rd());

        std::normal_distribution<InputType> input_distrib;
        for (unsigned int j = 0; j < input_amount; ++j)
            neuron.input_weight_.push_back(input_distrib(gen));
        std::normal_distribution<OutputType> output_distrib;
        neuron.output_weight_ = output_distrib(gen);
    }
}

template <typename InputType, typename OutputType>
typename NeuralNetwork<InputType, OutputType>::ForwardResult
NeuralNetwork<InputType, OutputType>::forward(InputsType inputs) const
{
    if (inputs.size() != input_amount_)
        throw std::invalid_argument("Inputs number different from expected");
    ForwardResult res;

    for (const auto& neuron : neurons_)
    {
        InputType hidden_sum = 0;
        for (unsigned int i = 0; i < input_amount_; ++i)
            hidden_sum += neuron.input_weight_[i] * inputs[i];
        InputType hidden_result = activate_(hidden_sum);
        res.hidden_sums_.push_back(hidden_sum);
        res.hidden_results_.push_back(hidden_result);
    }

    for (unsigned int i = 0; i < neurons_amount_; ++i)
        res.output_sum_ += res.hidden_results_[i] * neurons_[i].output_weight_;
    res.output_result_ = activate_(res.output_sum_);
    return res;
}

template <typename InputType, typename OutputType>
OutputType
NeuralNetwork<InputType, OutputType>::back_propagate(InputsType inputs,
                                                     OutputType target,
                                                     ForwardResult res)
{
    double learning_rate = 0.5;
    OutputType error = target - res.output_result_;
    OutputType delta_output_layer = activate_prime_(res.output_sum_) * error;
    // compute changes in neurons output weight
    InputsType hidden_output_changes;
    std::transform(res.hidden_results_.begin(), res.hidden_results_.end(),
                   std::back_inserter(hidden_output_changes),
                   [&delta_output_layer, &learning_rate](const auto& hidden_res)
                   { return (delta_output_layer / hidden_res) * learning_rate; });

    // compute delta hidden changes
    InputsType delta_hidden_layers;
    for (unsigned int i = 0; i < neurons_amount_; ++i)
        delta_hidden_layers.push_back((delta_output_layer
                                       / neurons_[i].output_weight_)
                                      * activate_prime_(res.hidden_sums_[i]));


    // update input weights
    for (unsigned int i = 0; i < input_amount_; ++i)
        for (unsigned int j = 0; j < neurons_amount_; ++j)
            neurons_[j].input_weight_[i] += delta_hidden_layers[j] / inputs[i];

    // update neurons output weight
    for (unsigned int i = 0; i < neurons_amount_; ++i)
        neurons_[i].output_weight_ += hidden_output_changes[i];

    return error;
}

template <typename InputType, typename OutputType>
void
NeuralNetwork<InputType, OutputType>::train(InputsType inputs,
                                            OutputType target)
{
    for (int i = 0; i < 1000; ++i)
    {
        auto res = forward(inputs);
        auto error = back_propagate(inputs, target, res);
        std::cout << error << std::endl;
    }
}

#pragma once

#include <functional>
#include <vector>

template <typename InputType, typename OutputType>
class NeuralNetwork
{
private:
    using InputsType = std::vector<InputType>;
    using OutputsType = std::vector<OutputType>;

    struct ForwardResult
    {
        InputsType hidden_sums_;
        InputsType hidden_results_;
        OutputType output_sum_;
        OutputType output_result_;
    };

    struct Neuron
    {
        InputsType input_weight_;
        OutputType output_weight_;
    };

public:
    NeuralNetwork(unsigned int input_amount, unsigned int output_amount,
                  unsigned int neurons_amount,
        const std::function<OutputType(InputType)>& activation_func,
        const std::function<OutputType(InputType)>& prime_activation_func);

    void train(InputsType inputs, OutputType target);

private:
    ForwardResult forward(InputsType inputs) const;
    OutputType back_propagate(InputsType inputs, OutputType target,
        ForwardResult res);

private:
    const unsigned int input_amount_;
    const unsigned int output_amount_;
    const unsigned int neurons_amount_;
    std::vector<Neuron> neurons_;
    const std::function<OutputType(InputType)> activate_;
    const std::function<OutputType(InputType)> activate_prime_;
};

#include "neural-network.hxx"

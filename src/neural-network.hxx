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
    : input_nb_(input_nb)
    , hidden_layer_(hidden_nb, Neuron<T>(input_nb + 1, activate)) // + 1 for the bias
    , output_layer_(output_nb, Neuron<T>(hidden_nb + 1, activate)) // + 1 for the bias
    , activate_(activate)
    , activate_prime_(activate_prime)
{
    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_real_distribution<> distrib(-1, 1);
    for (auto& n : hidden_layer_)
        for (auto& w : n.in_weights_)
            w = distrib(gen);

    for (auto& n : output_layer_)
        for (auto& w : n.in_weights_)
            w = distrib(gen);
}

template <typename T>
void NeuralNetwork<T>::feed_forward(std::vector<T> inputs)
{
    if (inputs.size() != input_nb_)
        throw std::invalid_argument("Incorrect inputs amount.");

    // feed hidden layer
    for (typename layer_type::size_type i = 0; i < hidden_layer_.size(); ++i)
        hidden_layer_[i].feed(inputs);

    // feed output layer
    std::vector<T> tmp_inputs(hidden_layer_.size());
    std::transform(hidden_layer_.begin(), hidden_layer_.end(),
                   tmp_inputs.begin(),
                   [](const auto& e){ return e.activated_output_; });
    for (typename layer_type::size_type i = 0; i < output_layer_.size(); ++i)
        output_layer_[i].feed(tmp_inputs);
}

template <typename T>
void NeuralNetwork<T>::back_propagate(std::vector<T> inputs, T target)
{
    double learning_rate = 0.17;
    std::vector<std::vector<T>> new_out_weights(output_layer_.size());
    std::vector<std::vector<T>> new_hidden_weights(hidden_layer_.size());

    // compute new weight for each output
    for (typename layer_type::size_type n = 0; n < output_layer_.size(); ++n)
    {
        auto delta_o1 = ((output_layer_[n].activated_output_ - target)
                         * output_layer_[n].activated_output_
                         * (1 - output_layer_[n].activated_output_));
        for (typename layer_type::size_type i = 0; i < hidden_layer_.size(); ++i)
        {
            auto delta_wi = delta_o1 * hidden_layer_[i].activated_output_;
            auto new_weight = (output_layer_[n].in_weights_[i] - delta_wi
                               * learning_rate);
            new_out_weights[n].push_back(new_weight);
        }
        auto new_bias_w = (output_layer_[n].in_weights_.back() - delta_o1
                           * learning_rate); // Bias
        new_out_weights[n].push_back(new_bias_w);
    }

    // compute new weight for each hidden node
    // E_w1 = E_outh1 * outh1_neth1 * neth1_w1
    for (typename layer_type::size_type i = 0; i < hidden_layer_.size(); ++i)
    {
        double Etotal_outh1 = 0.0;
        for (auto& o : output_layer_)
        {
            auto E_outo1 = (o.activated_output_ - target);
            auto outo1_neto1 = o.activated_output_ * (1 - o.activated_output_);
            auto neto1_outh1 = o.in_weights_[i];
            auto Eo1_outh1 = E_outo1 * outo1_neto1 * neto1_outh1;
            Etotal_outh1 += Eo1_outh1;
        }
        auto& h = hidden_layer_[i];
        auto outh1_neth1 = h.activated_output_ * (1 - h.activated_output_);

        // update each input weight of the neuron h
        for (typename layer_type::size_type j = 0; j < input_nb_; ++j)
        {
            auto neth1_w1 = inputs[j];
            auto delta_w1 = Etotal_outh1 * outh1_neth1 * neth1_w1;
            auto new_weight = h.in_weights_[j] - learning_rate * delta_w1;
            new_hidden_weights[i].push_back(new_weight);
        }
        auto new_bias_w = (h.in_weights_.back() - Etotal_outh1 * outh1_neth1
                           * learning_rate); // Bias
        new_hidden_weights[i].push_back(new_bias_w);
    }

    apply_new_weights(new_hidden_weights, new_out_weights);
}

template <typename T>
void
NeuralNetwork<T>::apply_new_weights(const std::vector<std::vector<T>>& hidden_w,
                                    const std::vector<std::vector<T>>& output_w)
{
    for (typename layer_type::size_type i = 0; i < hidden_layer_.size(); ++i)
        hidden_layer_[i].in_weights_ = hidden_w[i];

    for (typename layer_type::size_type i = 0; i < output_layer_.size(); ++i)
        output_layer_[i].in_weights_ = output_w[i];
}

template <typename T>
void NeuralNetwork<T>::train(const std::vector<T>& inputs, T target)
{
    for (int i = 0; i < 1000; ++i)
    {
        feed_forward(inputs);
        back_propagate(inputs, target);
    }
}

template <typename T>
T NeuralNetwork<T>::compute(const std::vector<T>& inputs)
{
    feed_forward(inputs);
    return output_layer_[0].activated_output_;
}

// Non member functions
template <typename T>
std::ostream& operator<<(std::ostream& ostr, const NeuralNetwork<T>& nn)
{
    ostr << "digraph neural_net" << std::endl
         << "{" << std::endl;

    for (typename NeuralNetwork<T>::layer_type::size_type i = 0;
         i < nn.hidden_layer_.size(); ++i)
    {
        const auto& n = nn.hidden_layer_[i];
        for (typename std::vector<T>::size_type j = 0; j < n.in_weights_.size() - 1;
             ++j)
        {
            ostr << "i" << j << " -> " << "\"" << n.activated_output_
                 << "\" [label=\"" << n.in_weights_[j] << "\" ];" << std::endl;
        }
    }

    for (typename NeuralNetwork<T>::layer_type::size_type i = 0;
         i < nn.output_layer_.size(); ++i)
    {
        const auto& n = nn.output_layer_[i];
        for (typename std::vector<T>::size_type j = 0; j < n.in_weights_.size() - 1;
             ++j)
        {
            ostr << "\"" << nn.hidden_layer_[j].activated_output_ << "\" -> \""
                 << n.activated_output_ << "\" [label=\""
                 << n.in_weights_[j] << "\" ];" << std::endl;
        }
    }

    return ostr << "}" << std::endl;
}

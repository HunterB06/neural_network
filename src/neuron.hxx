#pragma once

#include <algorithm>

template <typename T>
Neuron<T>::Neuron(unsigned int nb_inputs, const std::function<T(T)>& activate)
    : nb_inputs_(nb_inputs)
    , in_weights_(nb_inputs)
    , activate_(activate)
{}

template <typename T>
void Neuron<T>::feed(const std::vector<T>& inputs)
{
    output_ = 0;
    for (typename std::vector<T>::size_type i = 0; i < inputs.size(); ++i)
        output_ += inputs[i] * in_weights_[i];
    // add bias
    output_ += in_weights_.back();

    activated_output_ = activate_(output_);
}

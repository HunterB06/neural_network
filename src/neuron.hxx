#pragma once

#include <algorithm>

template <typename T>
Neuron<T>::Neuron(unsigned int nb_outputs, std::function<T(T)> activate)
    : nb_outputs_(nb_outputs)
    , out_weights_(nb_outputs)
    , outputs_(nb_outputs)
    , activated_outputs_(nb_outputs)
    , activate_(activate)
{}

template <typename T>
void Neuron<T>::feed(T input)
{
    for (typename std::vector<T>::size_type i = 0; i < outputs_.size(); ++i)
        outputs_[i] = input * out_weights_[i];

    std::transform(outputs_.begin(), outputs_.end(), activated_outputs_.begin(),
                   activate_);
}

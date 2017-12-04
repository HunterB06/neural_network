#pragma once

template <typename T>
Neuron<T>::Neuron(unsigned int nb_outputs)
    : nb_outputs_(nb_outputs)
    , out_weights_(nb_outputs)
    , outputs_(nb_outputs)
{}

template <typename T>
void Neuron<T>::feed(T input)
{
    for (typename std::vector<T>::size_type i = 0; i < outputs_.size(); ++i)
        outputs_[i] = input * out_weights_[i];
}

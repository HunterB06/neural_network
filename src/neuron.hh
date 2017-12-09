#pragma once

#include <vector>
#include <functional>

template <typename T>
class Neuron
{
public:
    Neuron(unsigned int nb_inputs, const std::function<T(T)>& activate);

    /**
    ** Give an input to the neuron and fill the outputs vector.
    */
    void feed(const std::vector<T>& input);

public:
    unsigned int nb_inputs_;
    std::vector<T> in_weights_;

    /**
    ** The outputs of the neuron before the activation function
    */
    T output_;

    /**
    ** The outputs of the neuron after the activation function
    */
    T activated_output_;
    const std::function<T(T)>& activate_;
};

#include "neuron.hxx"

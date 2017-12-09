#pragma once

#include <vector>
#include <functional>

template <typename T>
class Neuron
{
public:
    using input_type = std::vector<T>;

    Neuron(unsigned int nb_inputs, const std::function<T(T)>& activate);

    /**
    ** Give an input to the neuron and fill the outputs vector.
    */
    void feed(const input_type& input);

public:
    /**
    ** Number of inputs to the neuron. It corresponds to the number
    ** of neuron presents in the previous layer + the bias
    */
    unsigned int nb_inputs_;

    /**
    ** The weights for each connection to the previous layer. As nb_inputs,
    ** the size of the vector is the amount of neurons in the previous layer + 1
    ** for the bias
    */
    std::vector<T> in_weights_;

    /**
    ** The outputs of the neuron before the activation function
    */
    T output_;

    /**
    ** The outputs of the neuron after the activation function
    */
    T activated_output_;

    /**
    ** Reference to the activation function of the neural network.
    ** Don't need a copy since the Neuron doesn't have to live by itself.
    */
    const std::function<T(T)>& activate_;
};

#include "neuron.hxx"

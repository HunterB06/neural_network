#pragma once

#include <vector>

template <typename T>
class Neuron
{
public:
    Neuron(unsigned int nb_outputs);

    /**
    ** Give an input to the neuron and fill the outputs vector.
    */
    void feed(T input);

public:
    unsigned int nb_outputs_;
    std::vector<T> out_weights_;

    /**
    ** The outputs of the neuron before the activation function
    */
    std::vector<T> outputs_;

    /**
    ** The outputs of the neuron after the activation function
    */
    std::vector<T> activated_outputs_;
};

#include "neuron.hxx"

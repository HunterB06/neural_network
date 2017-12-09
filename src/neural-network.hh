#pragma once

#include <functional>
#include <vector>
#include <ostream>

#include "neuron.hh"

template <typename T>
class NeuralNetwork
{
public:
    using layer_type = std::vector<Neuron<T>>;
    using input_type = std::vector<T>;
    using training_set_type = std::vector<input_type>;
    using output_set = std::vector<T>;

    NeuralNetwork(unsigned int input_nb, unsigned int hidden_nb,
                  unsigned int output_nb, const std::function<T(T)>& activate);

    /**
    ** Train the network using expected output with corresponding inputs.
    ** @param inputs: a set of inputs that the neural net will learn to handle.
    ** @param target: the expected output corresponding to the inputs.
    ** @param iter_nb: the number of time the network will train itself
    */
    void train(const training_set_type& inputs, const output_set& target,
               unsigned int iter_nb);

    /**
    ** Compute the output of the neural network. This function should be
    ** efficient only after training
    */
    T compute(const input_type& inputs);
    T operator()(const input_type& inputs);

private:
    void feed_forward(const input_type& inputs);
    void back_propagate(const input_type& inputs, T target);
    void apply_new_weights(const std::vector<std::vector<T>>& hidden_w,
                           const std::vector<std::vector<T>>& output_w);

    template <typename U>
    friend std::ostream& operator<<(std::ostream& ostr,
                                    const NeuralNetwork<U>& n);
private:
    unsigned int input_nb_;
    layer_type hidden_layer_;
    layer_type output_layer_;

    const std::function<T(T)> activate_;
};

#include "neural-network.hxx"

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

    void train(const training_set_type& inputs, const output_set& target);
    T compute(const input_type& inputs);

private:
    void feed_forward(const input_type& inputs);
    void back_propagate(const input_type& inputs, T target);
    void apply_new_weights(const std::vector<std::vector<T>>& hidden_w,
                           const std::vector<std::vector<T>>& output_w);

    template <typename U>
    friend std::ostream& operator<<(std::ostream& ostr, const NeuralNetwork<U>& n);
private:
    unsigned int input_nb_;
    layer_type hidden_layer_;
    layer_type output_layer_;

    const std::function<T(T)> activate_;
};

#include "neural-network.hxx"

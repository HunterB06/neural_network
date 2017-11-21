#include <cmath>

#include "neural-network.hh"

static double sigmoid(double x)
{
    return 1 / (1 + std::exp(-x));
}

static double sigmoid_prime(double x)
{
    return std::exp(x) / std::pow(1 + std::exp(x), 2);
}

int main()
{
    NeuralNetwork<double, double> n(2, 1, 3, sigmoid, sigmoid_prime);
    n.train({1, 1}, 0);
    return 0;
}

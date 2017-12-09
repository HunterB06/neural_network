#include <cmath>
#include <fstream>

#include "neural-network.hh"

static double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

static double sigmoid_prime(double x)
{
    return std::exp(x) / std::pow(1 + std::exp(x), 2);
}

int main()
{
    NeuralNetwork<double> n(2, 3, 1, sigmoid, sigmoid_prime);
    std::cout << n.compute({1.0, 0.0}) << std::endl;
    n.train({1.0, 0.0}, 1.0);
    std::cout << n.compute({1.0, 0.0}) << std::endl;
    return 0;
}

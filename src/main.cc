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
    NeuralNetwork<double> n(2, 3, 1, sigmoid, sigmoid_prime);
//    n.feed_forward({1.0, 1.0});
    std::cout << n.compute({0.0, 1.0}) << std::endl;
    for (int i = 0; i < 1000; ++i)
        n.train({0.0, 1.0}, 1.0);
    std::cout << n.compute({0.0, 1.0}) << std::endl;
    return 0;
}

#include <cmath>
#include <fstream>

#include "neural-network.hh"

static double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

int main()
{
    NeuralNetwork<double> XOR(2, 3, 1, sigmoid);

    std::vector<double> ff{ 0.0, 0.0 };
    std::vector<double> tf{ 1.0, 0.0 };
    std::vector<double> ft{ 0.0, 1.0 };
    std::vector<double> tt{ 1.0, 1.0 };

    std::cout << XOR({1.0, 0.0}) << std::endl;

    XOR.train({ff, tf, ft, tt}, { 0.0, 1.0, 1.0, 0.0 }, 100000);

    std::cout << "1 XOR 1: " << XOR({1.0, 1.0}) << std::endl;
    std::cout << "0 XOR 1: " << XOR({0.0, 1.0}) << std::endl;
    std::cout << "1 XOR 0: " << XOR({1.0, 0.0}) << std::endl;
    std::cout << "0 XOR 0: " << XOR({0.0, 0.0}) << std::endl;

    return 0;
}

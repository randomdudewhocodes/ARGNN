#pragma once

#include "Layer.hpp"
#include <vector>

struct NeuralNetwork
{
	std::vector<Layer> layers;

	double learning_rate;
	double beta1;
	double beta2;
	double epsilon;
	int timestep;

	NeuralNetwork(const std::vector<int>& layer_sizes,
				  const Activation::ActivationFunction& activation,
				  double learning_rate = 0.001,
				  double beta1 = 0.9,
				  double beta2 = 0.999,
				  double epsilon = 1e-8);

	Matrix forward(const Matrix& input);

	void train(const Matrix& input, const Matrix& target, int epoch, int batch_size);

	Matrix predict(const Matrix& input);
};
#pragma once

#include "Matrix.hpp"
#include "Activation.hpp"

struct Layer
{
	Matrix weights;
	Matrix biases;
	Activation::ActivationFunction activation;

	Matrix input_cache;
	Matrix z_cache;

	Matrix dW;
	Matrix dB;

	Matrix m_weights;
	Matrix v_weights;
	Matrix m_biases;
	Matrix v_biases;

	Layer(int input_size, int output_size, Activation::ActivationFunction activation);

	Matrix forward(const Matrix& input);

	Matrix backward(const Matrix& dA);

	void updateParameters(int timestep, double learning_rate, double beta1, double beta2, double epsilon);
};
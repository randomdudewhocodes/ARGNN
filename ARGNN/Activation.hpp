#pragma once

#include <cmath>
#include <functional>

namespace Activation
{
	inline double sigmoid(double x) { return 1 / (1 + std::exp(-x)); }
	inline double sigmoid_derivative(double x)
	{
		auto s = sigmoid(x);
		return s * (1 - s);
	}

	inline double relu(double x) { return x > 0 ? x : 0; }
	inline double relu_derivative(double x) { return x > 0 ? 1 : 0; }

	inline double tanh(double x) { return std::tanh(x); }
	inline double tanh_derivative(double x)
	{
		auto t = tanh(x);
		return 1 - t * t;
	}

	struct ActivationFunction
	{
		std::function<double(double)> func;
		std::function<double(double)> derivative;
	};
}
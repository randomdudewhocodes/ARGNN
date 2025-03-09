#include "Layer.hpp"
#include <random>
#include <cmath>
#include <omp.h>

Layer::Layer(int input_size, int output_size, Activation::ActivationFunction act)
    : weights(output_size, input_size), biases(output_size, 1), activation(act),
      input_cache(0, 0), z_cache(0, 0), dW(output_size, input_size), dB(output_size, 1),
      m_weights(output_size, input_size), v_weights(output_size, input_size),
      m_biases(output_size, 1), v_biases(output_size, 1)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-1, 1);
    
    #pragma omp parallel for
    for (int idx = 0, total = weights.rows * weights.cols; idx < total; idx++)
        weights.data[idx] = dist(gen);
}

Matrix Layer::forward(const Matrix& input)
{
    input_cache = input;
    z_cache = (weights * input) + biases;
    
    Matrix a(z_cache.rows, z_cache.cols);
    #pragma omp parallel for
    for (int idx = 0, total = z_cache.rows * z_cache.cols; idx < total; idx++)
        a.data[idx] = activation.func(z_cache.data[idx]);
    
    return a;
}

Matrix Layer::backward(const Matrix &dA)
{
    Matrix dZ(z_cache.rows, z_cache.cols);
    #pragma omp parallel for
    for (int idx = 0, total = z_cache.rows * z_cache.cols; idx < total; idx++)
        dZ.data[idx] = dA.data[idx] * activation.derivative(z_cache.data[idx]);
    
    dW = dZ * input_cache.transpose();
    
    #pragma omp parallel for
    for (int i = 0; i < dZ.rows; i++)
    {
        double sum = 0;
        for (int j = 0; j < dZ.cols; j++)
            sum += dZ(i, j);
        dB(i, 0) = sum;
    }
    
    return weights.transpose() * dZ;
}

void Layer::updateParameters(int timestep, double learning_rate, double beta1, double beta2, double epsilon)
{
    double beta1_t = 1 - std::pow(beta1, timestep);
    double beta2_t = 1 - std::pow(beta2, timestep);
    
    #pragma omp parallel for
    for (int idx = 0, total = weights.rows * weights.cols; idx < total; idx++)
    {
        m_weights.data[idx] = beta1 * m_weights.data[idx] + (1 - beta1) * dW.data[idx];
        v_weights.data[idx] = beta2 * v_weights.data[idx] + (1 - beta2) * (dW.data[idx] * dW.data[idx]);
        weights.data[idx] -= learning_rate * (m_weights.data[idx] / beta1_t) / (std::sqrt(v_weights.data[idx] / beta2_t) + epsilon);
    }
    
    #pragma omp parallel for
    for (int idx = 0, total = biases.rows * biases.cols; idx < total; idx++)
    {
        m_biases.data[idx] = beta1 * m_biases.data[idx] + (1 - beta1) * dB.data[idx];
        v_biases.data[idx] = beta2 * v_biases.data[idx] + (1 - beta2) * (dB.data[idx] * dB.data[idx]);
        biases.data[idx] -= learning_rate * (m_biases.data[idx] / beta1_t) / (std::sqrt(v_biases.data[idx] / beta2_t) + epsilon);
    }
}

#include "NeuralNetwork.hpp"
#include <algorithm>
#include <random>
#include <omp.h>

NeuralNetwork::NeuralNetwork(const std::vector<int>& layer_sizes,
							 const Activation::ActivationFunction& activation,
	                         double learning_rate, double beta1, double beta2, double epsilon)
	: learning_rate(learning_rate), beta1(beta1), beta2(beta2), epsilon(epsilon), timestep(1)
{
	for (size_t i = 0; i < layer_sizes.size() - 1; i++)
		layers.emplace_back(layer_sizes[i], layer_sizes[i + 1], activation);
}

Matrix NeuralNetwork::forward(const Matrix& input)
{
	Matrix output = input;
	for (auto& layer : layers)
		output = layer.forward(output);

	return output;
}

static inline double computeMSE(const Matrix& output, const Matrix& target, int batch_samples)
{
    double sum = 0;
    int total = output.rows * output.cols;
    #pragma omp parallel for reduction(+:sum)
    for (int idx = 0; idx < total; idx++)
    {
        double diff = output.data[idx] - target.data[idx];
        sum += diff * diff;
    }
    return 0.5 * sum / batch_samples;
}

void NeuralNetwork::train(const Matrix& input, const Matrix& target, int epochs, int batch_size)
{
    int total_samples = input.cols;
    std::vector<int> indices(total_samples);
    for (int i = 0; i < total_samples; i++)
        indices[i] = i;
    
    std::random_device rd;
    std::mt19937 g(rd());
    
    Matrix mini_input(input.rows, batch_size);
    Matrix mini_target(target.rows, batch_size);
    
    for (int epoch = 1; epoch <= epochs; epoch++)
    {
        std::shuffle(indices.begin(), indices.end(), g);
        double epoch_loss = 0;
        int num_batches = 0;
        
        for (int start = 0; start < total_samples; start += batch_size)
        {
            int current_batch_size = std::min(batch_size, total_samples - start);
            
            for (int j = 0; j < current_batch_size; j++)
            {
                int idx = indices[start + j];
                for (int i = 0; i < input.rows; i++)
                    mini_input(i, j) = input(i, idx);
                
                for (int i = 0; i < target.rows; i++)
                    mini_target(i, j) = target(i, idx);
            }
            
            Matrix* current_input_ptr;
            Matrix* current_target_ptr;
            Matrix current_input(input.rows, current_batch_size);
            Matrix current_target(target.rows, current_batch_size);
            if (current_batch_size < batch_size)
            {
                for (int i = 0; i < input.rows; i++)
                    for (int j = 0; j < current_batch_size; j++)
                        current_input(i, j) = mini_input(i, j);

                for (int i = 0; i < target.rows; i++)
                    for (int j = 0; j < current_batch_size; j++)
                        current_target(i, j) = mini_target(i, j);

                current_input_ptr = &current_input;
                current_target_ptr = &current_target;
            }
            else
            {
                current_input_ptr = &mini_input;
                current_target_ptr = &mini_target;
            }
            
            Matrix output = forward(*current_input_ptr);
            
            double loss = computeMSE(output, *current_target_ptr, current_batch_size);
            epoch_loss += loss;
            
            Matrix dA(output.rows, output.cols);
            int out_total = output.rows * output.cols;
            #pragma omp parallel for
            for (int idx = 0; idx < out_total; idx++)
                dA.data[idx] = (output.data[idx] - current_target_ptr->data[idx]) / current_batch_size;
            
            for (int i = layers.size() - 1; i >= 0; i--)
                dA = layers[i].backward(dA);
            
            timestep++;
            #pragma omp parallel for
            for (int i = 0; i < static_cast<int>(layers.size()); i++)
                layers[i].updateParameters(timestep, learning_rate, beta1, beta2, epsilon);
            
            num_batches++;
        }
        
        std::cout << "Epoch " << epoch 
                    << " Average Loss: " << (epoch_loss / num_batches) 
                    << std::endl;
    }
}

Matrix NeuralNetwork::predict(const Matrix& input) { return forward(input); }
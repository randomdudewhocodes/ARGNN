#include "NeuralNetwork.hpp"
#include <cmath>
#include <iostream>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define __STDC_LIB_EXT1__
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

int main()
{
    int width, height, channels;
    unsigned char *data = stbi_load("image.jpg", &width, &height, &channels, 0);
    
    const int num_samples = width * height;
    const int epochs = 10000;
    const int batch_size = 100;

    Matrix input(2, num_samples);
    Matrix target(channels, num_samples);

    for (int j = 0; j < num_samples; j++)
    {
        input(0, j) = j / height;
        input(1, j) = j % height;
        for (int k = 0; k < channels; k++)
            target(k, j) = double(data[j + k]) / 256;
    }

    stbi_image_free(data);

    std::vector<int> layer_sizes = { 2, 20, 20, 20, 20, 20, 20, 20, channels };
    Activation::ActivationFunction activation = {Activation::tanh, Activation::tanh_derivative};

    NeuralNetwork net(layer_sizes, activation, 0.001);

    net.train(input, target, epochs, batch_size);

    std::vector<unsigned char> pixels(width * height * channels);

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            Matrix test_input(2, 1);
            test_input(0, 0) = x;
            test_input(1, 0) = y;

            Matrix prediction = net.predict(test_input);

            for (int k = 0; k < channels; k++)
                pixels[channels * (y * width + x) + k] = unsigned char(prediction(k, 0));
        }
    }

    if (stbi_write_png("NN slop.png", width, height, channels, pixels.data(), width * channels))
        std::cout << "Saved!";
    else
        std::cout << "You're not deserved to be a coder >:|";
}
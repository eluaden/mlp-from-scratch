#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <Toolkit.h>

enum ActivationFunction {
    RELU,
    SIGMOID,
    TANH,
    SOFTMAX
};

class Layer
{
private:
    int units; 
    double temperature = 3.0; // Used for softmax temperature scaling
    std::vector<std::vector<double>> z; 
    ActivationFunction activation_function;
    std::vector<std::vector<double>> weights; 
    std::vector<double> biases;
public:
    Layer(int units = 0, ActivationFunction activation_function = RELU);
    ~Layer();

    /**
     * Initializes the weights and biases for the layer.
     * @param input_size The number of inputs to the layer.
     */
    void initialize_params(int input_size);

    /**
     * Applies the forward pass of the layer.
     * @param inputs The input matrix to the layer.
     * @return The output vector after applying the layer's weights, biases, and activation function.
     */
    std::vector<std::vector<double>> operator()(const std::vector<std::vector<double>>& inputs);

    /**
     * Predicts the output for the given input data. Without weights and biases initialization.
     * its const version of the operator() method.
     * @param inputs The input data to predict.
     * @return The predicted output.
     */
    std::vector<std::vector<double>> predict(const std::vector<std::vector<double>>& inputs) const;

    /**
     * Applies the activation function to the linear combination of inputs.
     * @param z The linear combination of inputs.
     * @return The activated output.
     */
    std::vector<std::vector<double>> activation(const std::vector<std::vector<double>>& z) const;


    /**
     * Computes the derivative of the activation function.
     * @return The derivative of the activation function.
     */
    std::vector<std::vector<double>> activation_derivative();

    /**
     * Returns the activation function of the layer.
     * @return The activation function of the layer.
     */
    ActivationFunction get_activation_function() const;

    /**
     * Returns the weights of the layer.
     * @return The weights of the layer.
     */
    std::vector<std::vector<double>> get_weights() const;

    /**
     * Returns the biases of the layer.
     * @return The biases of the layer.
     */
    std::vector<double> get_biases() const;

    /**
     * Modify the weights of the layer.
     * @param new_weights The new weights to set for the layer.
     */
    void set_weights(const std::vector<std::vector<double>>& new_weights);

    /**
     * Modify the biases of the layer.
     * @param new_biases The new biases to set for the layer.
     */
    void set_biases(const std::vector<double>& new_biases);

    /**
     * Outputs the layer's details to a stream.
     * @param os The output stream to write to.
     * @return The output stream after writing the layer's details.
     */
    std::ostream& operator<<(std::ostream& os) const;

};
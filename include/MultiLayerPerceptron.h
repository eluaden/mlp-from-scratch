#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <functional>
#include <omp.h>
#include "Layer.h"

enum LossFunction {
    BCE, // Binary Cross-Entropy
    MSE, // Mean Squared Error
    CCE // Categorical Cross-Entropy
};

class MultiLayerPerceptron {
private:
    std::vector<Layer> layers; 
    std::vector<std::vector<std::vector<double>>> activations; // Store activations for each layer
    LossFunction loss_type; 
    std::function<double (const std::vector<std::vector<double>>&, const std::vector<std::vector<double>>&)> loss_function; // Loss function
    std::function<std::vector<std::vector<double>>(const std::vector<std::vector<double>>&, const std::vector<std::vector<double>>)> loss_derivative; // Derivative of the loss function
    std::vector<double> X_mean; 
    std::vector<double> X_std; 

public:


    MultiLayerPerceptron(std::vector<Layer> layers = {});

    ~MultiLayerPerceptron();

    /**
     * predicts the output for the given input data.
     * @param input The input data to predict.
     * @return The predicted output.
     */
    std::vector<std::vector<double>> predict(const std::vector<std::vector<double>>& input) const;


    /**
     * forwards the input through the network.
     * @param input The input data to forward.
     * @return The output after the forward pass.
     */
    std::vector<std::vector<double>> forward(const std::vector<std::vector<double>>& input);

    /**
     * backwards the error through the network.
     * @param input The input data to backpropagate.
     * @param expected_output The expected output to compare against.
     * @param learning_rate The learning rate for the optimizer.
     */
    void backward(const std::vector<std::vector<double>> input ,const std::vector<std::vector<double>>& expected_output, double learning_rate = 0.01);

    /**
     * compiles the model with the given loss function and optimizer.
     * @param loss_type The loss function to use.
     */
    void compile(LossFunction loss_type);

    /**
     * fits the model to the training data.
     * @param training_data The training data to fit the model to.
     * @param expected_output The expected output for the training data.
     * @param epochs The number of epochs to train for.
     * @param learning_rate The learning rate for the optimizer.
     * @param batch_size The size of the batches to use for training.
     */

    void fit(const std::vector<std::vector<double>>& training_data, 
             const std::vector<std::vector<double>>& expected_output, 
             int epochs = 100, 
             double learning_rate = 0.01, 
             int batch_size = 32);

    /**
     * Calculates the accuracy of the model on the given test data.
     * @param test_data The test data to evaluate the model on.
     * @param expected_output The expected output for the test data.
     * @return The accuracy of the model as a percentage.
     */
    double accuracy(const std::vector<std::vector<double>>& test_data, 
                    const std::vector<std::vector<double>>& expected_output) const;
    
    /**
     * Outputs the model's details to a stream.
     * @param os The output stream to write to.
     * @return The output stream after writing the model's details.
     */
    std::ostream& operator<<(std::ostream& os) const;
    

    

    
};



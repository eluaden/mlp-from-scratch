#include "MultiLayerPerceptron.h"

MultiLayerPerceptron::MultiLayerPerceptron(std::vector<Layer> layers)
    : layers(std::move(layers)){};

MultiLayerPerceptron::~MultiLayerPerceptron() {}


std::vector<std::vector<double>> MultiLayerPerceptron::predict(const std::vector<std::vector<double>>& input) const {
    
    std::vector<std::vector<double>> output;
    std::vector<std::vector<double>> normalized_input = input;
    //normalize the input data

    for(size_t i = 0; i < normalized_input.size(); ++i) {
        normalized_input[i] = tk::elementwise_operation(normalized_input[i], X_mean, '-');
        normalized_input[i] = tk::elementwise_operation(normalized_input[i], X_std, '/');
    }

    output = normalized_input; 

    for (auto& layer : layers) {
        output = layer.predict(output); // Use the predict method of each layer
    }


    if (layers.empty()) {
        throw std::runtime_error("No layers in the model. Please add layers before predicting.");
    }
    if (output.empty() || output[0].empty()) {
        throw std::runtime_error("Output is empty. Please check the input data.");
    }

    return output;
}


std::vector<std::vector<double>> MultiLayerPerceptron::forward(const std::vector<std::vector<double>>& input) {
    std::vector<std::vector<double>> output = input;
    activations.clear(); // Clear previous activations
        activations.push_back(input); // Store the input as the first activation


    for (auto& layer : layers) {
        output = layer(output);
        activations.push_back(output); // Store the activation for each layer
    }

    return output;
}

void MultiLayerPerceptron::backward(const std::vector<std::vector<double>> input ,const std::vector<std::vector<double>>& expected_output, double learning_rate) 
{
        int m = input.size();

    std::vector<std::vector<double>> predict = forward(input);
    std::vector<std::vector<double>> loss = loss_derivative(predict, expected_output);

    for (int i = layers.size() - 1; i >= 0; --i) 
    {
        Layer& layer = layers[i];
        std::vector<std::vector<double>> activation = activations[i];

        //dw = (activation^T * loss) / m
        std::vector<std::vector<double>> dw = tk::elementwise_operation(
            tk::matmul(tk::transpose(activation), loss), 
            static_cast<double>(m), 
            '/'
        );
        
        //db = sum(loss, 0) / m
        std::vector<double> db = tk::elementwise_operation(
            tk::sum(loss, 0), 
            static_cast<double>(m), 
            '/'
        );

        std::vector<std::vector<double>> new_weights = layer.get_weights();
        std::vector<double> new_biases = layer.get_biases();

        dw = tk::elementwise_operation(dw, learning_rate, '*');
        db = tk::elementwise_operation(db, learning_rate, '*');

        new_weights = tk::elementwise_operation(new_weights, dw, '-');
        new_biases = tk::elementwise_operation(new_biases, db, '-');

        layer.set_weights(new_weights);
        layer.set_biases(new_biases);

        if (i > 0)
        {
            Layer prev_layer = layers[i - 1];
            loss = tk::matmul(loss, tk::transpose(layer.get_weights()));
            loss = tk::elementwise_operation(loss, prev_layer.activation_derivative(), '*');
        }
        
    }
}

void MultiLayerPerceptron::compile(LossFunction loss_type) 
{
    this->loss_type = loss_type;
    switch (loss_type)
    {
    case MSE:
        loss_function = tk::mean_squared_error;
        loss_derivative = tk::mean_squared_error_derivative;
        break;
    case BCE:
        loss_function = tk::binary_cross_entropy;
        loss_derivative = tk::binary_cross_entropy_derivative;
        break;
    case CCE:
        loss_function = tk::categorical_cross_entropy;
        loss_derivative = tk::categorical_cross_entropy_derivative;
        break;
    
    }
}

void MultiLayerPerceptron::fit(const std::vector<std::vector<double>>& training_data, 
                               const std::vector<std::vector<double>>& expected_output, 
                               int epochs, 
                               double learning_rate, 
                               int batch_size) 
{
    if (training_data.empty() || expected_output.empty()) {
        throw std::invalid_argument("Training data and expected output cannot be empty.");
    }
    if (training_data.size() != expected_output.size()) {
        throw std::invalid_argument("Training data and expected output must have the same number of samples.");
    }

    //normalize the input data

    X_mean = tk::mean(training_data, 0);
    X_std = tk::std(training_data, 0);

    std::vector<std::vector<double>> normalized_data = training_data;

    for(size_t i = 0; i < normalized_data.size(); ++i) {
        normalized_data[i] = tk::elementwise_operation(normalized_data[i], X_mean, '-');
        normalized_data[i] = tk::elementwise_operation(normalized_data[i], X_std, '/');
    }

    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (size_t i = 0; i < normalized_data.size(); i += batch_size) {
            size_t end = std::min(i + batch_size, normalized_data.size());
            std::vector<std::vector<double>> batch_input(normalized_data.begin() + i, normalized_data.begin() + end);
            std::vector<std::vector<double>> batch_output(expected_output.begin() + i, expected_output.begin() + end);
            forward(batch_input);
            backward(batch_input, batch_output, learning_rate);

            
        }

        //print progress
        std::vector<std::vector<double>> predictions = predict(training_data);
        double loss = loss_function(predictions, expected_output);
        double acc = accuracy(training_data, expected_output);
        std::cout << "Epoch " << epoch << "| Loss: " << loss << "| Accuracy: " << acc * 100 << "%" << std::endl;

    }
}

double MultiLayerPerceptron::accuracy(const std::vector<std::vector<double>>& test_data, 
                                       const std::vector<std::vector<double>>& expected_output) const {
    if (test_data.empty() || expected_output.empty()) {
        throw std::invalid_argument("Test data and expected output cannot be empty.");
    }
    if (test_data.size() != expected_output.size()) {
        throw std::invalid_argument("Test data and expected output must have the same number of samples.");
    }

    int correct_predictions = 0;
    
    for (size_t i = 0; i < test_data.size(); ++i) {
        std::vector<std::vector<double>> prediction = predict({test_data[i]});

        // Find the index of the maximum value in the prediction
        auto max_it = std::max_element(prediction[0].begin(), prediction[0].end());
        int predicted_class = std::distance(prediction[0].begin(), max_it);

        // Find the index of the maximum value in the expected output
        auto expected_max_it = std::max_element(expected_output[i].begin(), expected_output[i].end());
        int expected_class = std::distance(expected_output[i].begin(), expected_max_it);

        if (predicted_class == expected_class) {
            correct_predictions++;
        }
    }


    return static_cast<double>(correct_predictions) / test_data.size(); // Return accuracy as a percentage
}





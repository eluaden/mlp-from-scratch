#include "Layer.h"


Layer::Layer(int units, ActivationFunction activation_function)
    : units(units), activation_function(activation_function) {}
    
Layer::~Layer() {}


void Layer::initialize_params(int input_size) 
{
    std::random_device rd;
    std::mt19937 gen(rd());

    double stddev = std::sqrt(2.0 / input_size);
    std::normal_distribution<> dis(0.0, stddev);

    weights.resize(input_size);
    biases.resize(units);

    for (int i = 0; i < input_size; ++i) {
        weights[i].resize(units);
        for (int j = 0; j < units; ++j) {
            weights[i][j] = dis(gen); // Initialize weights with a normal distribution
        }
    }

    for (int j = 0; j < units; ++j) {
        biases[j] = dis(gen); // Initialize biases with a normal distribution
    }
}


std::vector<std::vector<double>> Layer::operator()(const std::vector<std::vector<double>>& inputs) 
{
    int batch_size = inputs.size();
    int input_size = inputs[0].size();
    std::vector<std::vector<double>> outputs(inputs.size(), std::vector<double>(units, 0.0f));

    //lazy initialization of weights and biases
    if(weights.empty() || biases.empty())
    {
        initialize_params(input_size);
    }

    z = tk::matmul(inputs, weights); 
    for (int i = 0; i < batch_size; ++i) 
    {
        for (int j = 0; j < units; ++j) 
        {
            z[i][j] += biases[j]; 
        }
    }

    outputs = activation(z); 

    return outputs;
}


std::vector<std::vector<double>> Layer::predict(const std::vector<std::vector<double>>& inputs) const 
{
    if (weights.empty() || biases.empty()) 
    {
        throw std::runtime_error("Weights and biases must be initialized before prediction.");
    }

    std::vector<std::vector<double>> outputs(inputs.size(), std::vector<double>(units, 0.0f));
    std::vector<std::vector<double>> z = tk::matmul(inputs, weights); 

    for (size_t i = 0; i < inputs.size(); ++i) 
    {
        for (size_t j = 0; j < static_cast<size_t>(units); ++j) 
        {
            z[i][j] += biases[j]; 
        }
    }

    outputs = activation(z); 

    return outputs;
}

std::vector<std::vector<double>> Layer::activation(const std::vector<std::vector<double>>& z) const
{
    std::vector<std::vector<double>> activated(z.size(), std::vector<double>(units, 0.0));

    for (size_t i = 0; i < z.size(); ++i) {
        for (size_t j = 0; j < static_cast<size_t>(units); ++j) 
        {
            if (activation_function == RELU) 
            {
                activated[i][j] = std::max(0.0, z[i][j]); // ReLU activation
            } 
            else if (activation_function == SIGMOID) 
            {
                activated[i][j] = 1.0 / (1.0 + std::exp(-z[i][j])); // Sigmoid activation
            } 
            else if (activation_function == TANH) 
            {
                activated[i][j] = std::tanh(z[i][j]); // Tanh activation
            }
            else if (activation_function == SOFTMAX) 
            {

                double sum = 0.0;
                for (size_t k = 0; k < static_cast<size_t>(units); ++k) {
                    sum += std::exp(z[i][k] / temperature);
                }
                activated[i][j] = std::exp((z[i][j])/ temperature) / sum; // Softmax activation with temperature scaling
            } 
            else 
            {
                throw std::invalid_argument("Unknown activation function");
            }
            
            
        }
    }

    return activated;
}

std::vector<std::vector<double>> Layer::activation_derivative() 
{
    std::vector<std::vector<double>> derivative(z.size(), std::vector<double>(units, 0.0));

    for (size_t i = 0; i < z.size(); ++i) {
        for (size_t j = 0; j < static_cast<size_t>(units); ++j) 
        {
            if (activation_function == RELU) 
            {
                derivative[i][j] = (z[i][j] > 0) ? 1.0 : 0.0; // Derivative of ReLU
            } 
            else if (activation_function == SIGMOID) 
            {
                double sig = 1.0 / (1.0 + std::exp(-z[i][j]));
                derivative[i][j] = sig * (1 - sig); // Derivative of Sigmoid
            } 
            else if (activation_function == TANH) 
            {
                derivative[i][j] = 1 - std::tanh(z[i][j]) * std::tanh(z[i][j]); // Derivative of Tanh
            }
            else if (activation_function == SOFTMAX) 
            {
                throw std::invalid_argument("Softmax does not have a simple derivative. Use cross-entropy loss for backpropagation.");
            } 
            else 
            {
                throw std::invalid_argument("Unknown activation function");
            }
            

        }
    }

    

    return derivative;
}

ActivationFunction Layer::get_activation_function() const 
{
    return activation_function;
}

std::vector<std::vector<double>> Layer::get_weights() const 
{
    return weights;
}

std::vector<double> Layer::get_biases() const 
{
    return biases;
}

void Layer::set_weights(const std::vector<std::vector<double>>& new_weights) 
{
    if (new_weights.size() != weights.size() || new_weights[0].size() != static_cast<size_t>(units)) 
    {
        throw std::invalid_argument("New weights dimensions do not match the layer's dimensions.");
    }
    weights = new_weights;
}

void Layer::set_biases(const std::vector<double>& new_biases) 
{
    if (new_biases.size() != static_cast<size_t>(units)) 
    {
        throw std::invalid_argument("New biases size does not match the number of units in the layer.");
    }
    biases = new_biases;
}

std::ostream& Layer::operator<<(std::ostream& os) const
{
    os << "Layer with " << units << " units and activation function: ";
    switch (activation_function) 
    {
        case RELU: os << "ReLU"; break;
        case SIGMOID: os << "Sigmoid"; break;
        case TANH: os << "Tanh"; break;
        case SOFTMAX: os << "Softmax"; break;
        default: os << "Unknown";
    }
    os << "\nWeights:\n";
    for (const auto& row : weights) 
    {
        for (const auto& weight : row) 
        {
            os << weight << " ";
        }
        os << "\n";
    }
    os << "Biases:\n";
    for (const auto& bias : biases) 
    {
        os << bias << " ";
    }
    os << "\n";
    return os;
}
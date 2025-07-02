//This file contains the OpenMP implementation of the Toolkit library.
// The only parallelized function is the matrix multiplication (matmul).
// The rest of the functions are serial implementations, because they are not computationally intensive enough to benefit from parallelization.
#include "Toolkit.h"
#include <omp.h>

namespace tk {

std::vector<std::vector<double>> matmul(std::vector<std::vector<double>> a, std::vector<std::vector<double>> b) 
{
    if (a.empty() || b.empty()) 
    {
        throw std::invalid_argument("Input matrices cannot be empty.");
    }
    
    int a_rows = a.size();
    int a_cols = a[0].size();
    int b_rows = b.size();
    int b_cols = b[0].size();

    if (a_cols != b_rows) 
    {
        throw std::invalid_argument("Number of columns in the first matrix must match the number of rows in the second matrix.");
    }

    std::vector<std::vector<double>> result(a_rows, std::vector<double>(b_cols, 0.0));

    #pragma omp parallel for num_threads(8) collapse(2)
    for (int i = 0; i < a_rows; ++i) 
    {
        for (int j = 0; j < b_cols; ++j) 
        {
            for (int k = 0; k < a_cols; ++k) 
            {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }

    return result;

}

std::vector<std::vector<double>> transpose(const std::vector<std::vector<double>>& matrix) 
{
    if (matrix.empty()) 
    {
        throw std::invalid_argument("Input matrix cannot be empty.");
    }

    int rows = matrix.size();
    int cols = matrix[0].size();

    std::vector<std::vector<double>> transposed(cols, std::vector<double>(rows, 0.0));

    for (int i = 0; i < rows; ++i) 
    {
        for (int j = 0; j < cols; ++j) 
        {
            transposed[j][i] = matrix[i][j];
        }
    }

    return transposed;

}

std::vector<double> transpose(const std::vector<double>& vector) 
{
    if (vector.empty()) 
    {
        throw std::invalid_argument("Input vector cannot be empty.");
    }

    return std::vector<double>(vector.begin(), vector.end());
}


std::vector<double> sum(const std::vector<std::vector<double>>& matrix, int axis) 
{
    if (matrix.empty()) 
    {
        throw std::invalid_argument("Input matrix cannot be empty.");
    }

    if (axis != 0 && axis != 1) 
    {
        throw std::invalid_argument("Axis must be 0 (rows) or 1 (columns).");
    }

    int rows = matrix.size();
    int cols = matrix[0].size();
    std::vector<double> result;

    if (axis == 0) 
    { 
        result.resize(cols, 0.0);
        for (int j = 0; j < cols; ++j) 
        {
            for (int i = 0; i < rows; ++i) 
            {
                result[j] += matrix[i][j];
            }
        }
    } 
    else 
    { 
        result.resize(rows, 0.0);
        for (int i = 0; i < rows; ++i) 
        {
            for (int j = 0; j < cols; ++j) 
            {
                result[i] += matrix[i][j];
            }
        }
    }

    return result;
}

std::vector<std::vector<double>> elementwise_operation(const std::vector<std::vector<double>>& matrix, double scalar, char operation) 
{
    if (matrix.empty()) 
    {
        throw std::invalid_argument("Input matrix cannot be empty.");
    }

    std::vector<std::vector<double>> result(matrix.size(), std::vector<double>(matrix[0].size(), 0.0));

    #pragma omp parallel for num_threads(8) 
    for (size_t i = 0; i < matrix.size(); ++i) 
    {
        for (size_t j = 0; j < matrix[i].size(); ++j) 
        {
            if (operation == '+') 
            {
                result[i][j] = matrix[i][j] + scalar;
            } 
            else if (operation == '-') 
            {
                result[i][j] = matrix[i][j] - scalar;
            } 
            else if (operation == '*') 
            {
                result[i][j] = matrix[i][j] * scalar;
            } 
            else if (operation == '/') 
            {
                if (scalar == 0) 
                {
                    throw std::invalid_argument("Division by zero is not allowed.");
                }
                result[i][j] = matrix[i][j] / scalar;
            }
            else 
            {
                throw std::invalid_argument("Unknown operation. Use '+', '-', '*', or '/'.");
            }
        }
    }

    return result;
}

std::vector<double> elementwise_operation(const std::vector<double>& vector, double scalar, char operation) 
{
    std::vector<double> result(vector.size(), 0.0);

    for (size_t i = 0; i < vector.size(); ++i) 
    {
        if (operation == '+') 
        {
            result[i] = vector[i] + scalar;
        } 
        else if (operation == '-') 
        {
            result[i] = vector[i] - scalar;
        } 
        else if (operation == '*') 
        {
            result[i] = vector[i] * scalar;
        } 
        else if (operation == '/') 
        {
            if (scalar == 0) 
            {
                throw std::invalid_argument("Division by zero is not allowed.");
            }
            result[i] = vector[i] / scalar;
        }
        else 
        {
            throw std::invalid_argument("Unknown operation. Use '+', '-', '*', or '/'.");
        }
    }

    return result;

} 

std::vector<std::vector<double>> elementwise_operation(const std::vector<std::vector<double>>& a, const std::vector<std::vector<double>>& b, char operation) 
{
    if (a.empty() || b.empty()) 
    {
        throw std::invalid_argument("Input matrices cannot be empty.");
    }

    if (a.size() != b.size() || a[0].size() != b[0].size()) 
    {
        throw std::invalid_argument("Matrices must have the same dimensions for element-wise operations.");
    }

    std::vector<std::vector<double>> result(a.size(), std::vector<double>(a[0].size(), 0.0));

    #pragma omp parallel for num_threads(8)
    for (size_t i = 0; i < a.size(); ++i) 
    {
        for (size_t j = 0; j < a[i].size(); ++j) 
        {
            if (operation == '+') 
            {
                result[i][j] = a[i][j] + b[i][j];
            } 
            else if (operation == '-') 
            {
                result[i][j] = a[i][j] - b[i][j];
            } 
            else if (operation == '*') 
            {
                result[i][j] = a[i][j] * b[i][j];
            } 
            else if (operation == '/') 
            {
                if (b[i][j] == 0) 
                {
                    throw std::invalid_argument("Division by zero is not allowed.");
                }
                result[i][j] = a[i][j] / b[i][j];
            } 
            else 
            {
                throw std::invalid_argument("Unknown operation. Use '+', '-', '*', or '/'.");
            }
        }
    }

    return result;

}

std::vector<double> elementwise_operation(const std::vector<double>& a, const std::vector<double>& b, char operation) 
{
    if (a.size() != b.size()) 
    {
        throw std::invalid_argument("Vectors must have the same size for element-wise operations.");
    }

    std::vector<double> result(a.size(), 0.0);

    for (size_t i = 0; i < a.size(); ++i) 
    {
        if (operation == '+') 
        {
            result[i] = a[i] + b[i];
        } 
        else if (operation == '-') 
        {
            result[i] = a[i] - b[i];
        } 
        else if (operation == '*') 
        {
            result[i] = a[i] * b[i];
        }  
        else if (operation == '/') 
        {
            if (b[i] == 0) 
            {
                throw std::invalid_argument("Division by zero is not allowed.");
            }
            result[i] = a[i] / b[i];
        }
        else 
        {
            throw std::invalid_argument("Unknown operation. Use '+', '-', '*', or '/'.");
        }
    }

    return result;

}

std::vector<std::vector<double>> log(const std::vector<std::vector<double>>& matrix) 
{
    if (matrix.empty()) 
    {
        throw std::invalid_argument("Input matrix cannot be empty.");
    }

    std::vector<std::vector<double>> result(matrix.size(), std::vector<double>(matrix[0].size(), 0.0));

    for (size_t i = 0; i < matrix.size(); ++i) 
    {
        for (size_t j = 0; j < matrix[i].size(); ++j) 
        {
            if (matrix[i][j] <= 0) 
            {
                throw std::invalid_argument("Logarithm undefined for non-positive values.");
            }
            result[i][j] = std::log(matrix[i][j]);
        }
    }

    return result;

}

double mean_squared_error(const std::vector<std::vector<double>>& predictions, const std::vector<std::vector<double>>& targets) 
{
    if (predictions.empty() || targets.empty()) 
    {
        throw std::invalid_argument("Predictions and targets cannot be empty.");
    }

    if (predictions.size() != targets.size() || predictions[0].size() != targets[0].size()) 
    {
        throw std::invalid_argument("Predictions and targets must have the same dimensions.");
    }

    double mse = 0.0;
    size_t n = predictions.size();
    size_t m = predictions[0].size();

    for (size_t i = 0; i < n; ++i) 
    {
        for (size_t j = 0; j < m; ++j) 
        {
            mse += std::pow(predictions[i][j] - targets[i][j], 2);
        }
    }

    return mse / (n * m);
}

std::vector<std::vector<double>> mean_squared_error_derivative(const std::vector<std::vector<double>>& predictions, const std::vector<std::vector<double>>& targets) 
{
    if (predictions.empty() || targets.empty()) 
    {
        throw std::invalid_argument("Predictions and targets cannot be empty.");
    }

    if (predictions.size() != targets.size() || predictions[0].size() != targets[0].size()) 
    {
        throw std::invalid_argument("Predictions and targets must have the same dimensions.");
    }

    std::vector<std::vector<double>> mse_derivative(predictions.size(), std::vector<double>(predictions[0].size(), 0.0));

    for (size_t i = 0; i < predictions.size(); ++i) 
    {
        for (size_t j = 0; j < predictions[i].size(); ++j) 
        {
            mse_derivative[i][j] =(predictions[i][j] - targets[i][j]) / predictions[i].size();
        }
    }

    return mse_derivative;

}

double binary_cross_entropy(const std::vector<std::vector<double>>& predictions, const std::vector<std::vector<double>>& targets) 
{
    if (predictions.empty() || targets.empty()) 
    {
        throw std::invalid_argument("Predictions and targets cannot be empty.");
    }

    if (predictions.size() != targets.size() || predictions[0].size() != targets[0].size()) 
    {
        throw std::invalid_argument("Predictions and targets must have the same dimensions.");
    }

    double bce = 0.0;
    size_t n = predictions.size();
    size_t m = predictions[0].size();

    for (size_t i = 0; i < n; ++i) 
    {
        for (size_t j = 0; j < m; ++j) 
        {
            if (predictions[i][j] <= 0 || predictions[i][j] >= 1) 
            {   
                std::cout<<predictions[i][j]<<std::endl;
                throw std::invalid_argument("Predictions must be probabilities between 0 and 1.");
            }
            bce += -targets[i][j] * std::log(predictions[i][j]) - (1 - targets[i][j]) * std::log(1 - predictions[i][j]);
        }
        bce /= m;
    }

    return bce;

}

std::vector<std::vector<double>> binary_cross_entropy_derivative(const std::vector<std::vector<double>>& predictions, const std::vector<std::vector<double>>& targets) 
{
    if (predictions.empty() || targets.empty()) 
    {
        throw std::invalid_argument("Predictions and targets cannot be empty.");
    }

    if (predictions.size() != targets.size() || predictions[0].size() != targets[0].size()) 
    {
        throw std::invalid_argument("Predictions and targets must have the same dimensions.");
    }

    std::vector<std::vector<double>> bce_derivative(predictions.size(), std::vector<double>(predictions[0].size(), 0.0));

    bce_derivative = elementwise_operation(predictions, targets, '-');

    return bce_derivative;

}

double categorical_cross_entropy(const std::vector<std::vector<double>>& predictions, const std::vector<std::vector<double>>& targets) 
{
    if (predictions.empty() || targets.empty()) 
    {
        throw std::invalid_argument("Predictions and targets cannot be empty.");
    }

    if (predictions.size() != targets.size() || predictions[0].size() != targets[0].size()) 
    {
        throw std::invalid_argument("Predictions and targets must have the same dimensions.");
    }

    double cce = 0.0;
    size_t n = predictions.size();
    size_t m = predictions[0].size();


    for (size_t i = 0; i < n; ++i) 
    {
        for (size_t j = 0; j < m; ++j) 
        {
            if (predictions[i][j] <= 0 || predictions[i][j] >= 1) 
            {
                std::cout<<predictions[i][j]<<std::endl;
                throw std::invalid_argument("Predictions must be probabilities between 0 and 1.");
            }
            cce += -targets[i][j] * std::log(predictions[i][j]);
        }
        cce /= m;
    }


    return cce;

}

std::vector<std::vector<double>> categorical_cross_entropy_derivative(const std::vector<std::vector<double>>& predictions, const std::vector<std::vector<double>>& targets) 
{
    if (predictions.empty() || targets.empty()) 
    {
        throw std::invalid_argument("Predictions and targets cannot be empty.");
    }

    if (predictions.size() != targets.size() || predictions[0].size() != targets[0].size()) 
    {
        throw std::invalid_argument("Predictions and targets must have the same dimensions.");
    }

    std::vector<std::vector<double>> cce_derivative(predictions.size(), std::vector<double>(predictions[0].size(), 0.0));

    cce_derivative = elementwise_operation(predictions, targets, '-');


    return cce_derivative;
}

std::vector<double> mean(const std::vector<std::vector<double>>& matrix, int axis)
{
    if (matrix.empty()) 
    {
        throw std::invalid_argument("Input matrix cannot be empty.");
    }

    if (axis != 0 && axis != 1) 
    {
        throw std::invalid_argument("Axis must be 0 (rows) or 1 (columns).");
    }

    int rows = matrix.size();
    int cols = matrix[0].size();
    std::vector<double> result;

    if (axis == 0) 
    { 
        result.resize(cols, 0.0);
        for (int j = 0; j < cols; ++j) 
        {
            for (int i = 0; i < rows; ++i) 
            {
                result[j] += matrix[i][j];
            }
            result[j] /= rows;
        }
    } 
    else 
    { 
        result.resize(rows, 0.0);
        for (int i = 0; i < rows; ++i) 
        {
            for (int j = 0; j < cols; ++j) 
            {
                result[i] += matrix[i][j];
            }
            result[i] /= cols;
        }
    }

    return result;
}

std::vector<double> std(const std::vector<std::vector<double>>& matrix, int axis)
{
    if (matrix.empty()) 
    {
        throw std::invalid_argument("Input matrix cannot be empty.");
    }

    if (axis != 0 && axis != 1) 
    {
        throw std::invalid_argument("Axis must be 0 (rows) or 1 (columns).");
    }

    int rows = matrix.size();
    int cols = matrix[0].size();
    std::vector<double> result;

    if (axis == 0) 
    { 
        result.resize(cols, 0.0);
        for (int j = 0; j < cols; ++j) 
        {
            double mean = 0.0;
            for (int i = 0; i < rows; ++i) 
            {
                mean += matrix[i][j];
            }
            mean /= rows;

            for (int i = 0; i < rows; ++i) 
            {
                result[j] += std::pow(matrix[i][j] - mean, 2);
            }
            result[j] = std::sqrt(result[j] / rows);
        }
    } 
    else 
    { 
        result.resize(rows, 0.0);
        for (int i = 0; i < rows; ++i) 
        {
            double mean = 0.0;
            for (int j = 0; j < cols; ++j) 
            {
                mean += matrix[i][j];
            }
            mean /= cols;

            for (int j = 0; j < cols; ++j) 
            {
                result[i] += std::pow(matrix[i][j] - mean, 2);
            }
            result[i] = std::sqrt(result[i] / cols);
        }
    }

    return result;
}

}// namespace tk



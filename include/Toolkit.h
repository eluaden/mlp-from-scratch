#pragma once

#include <vector>
#include <iostream>
#include <stdexcept>
#include <cmath>

namespace tk {

    /**
     * Multiplies two matrices using the standard matrix multiplication algorithm.
     * @param a The first matrix.
     * @param b The second matrix.
     * @return The resulting matrix after multiplication.
     * @throws std::invalid_argument if the number of columns in the first matrix does not match the number of rows in the second matrix.
     */
    std::vector<std::vector<double>> matmul(std::vector<std::vector<double>> a, std::vector<std::vector<double>> b); 


    /**
     * Transposes a given matrix.
     * @param matrix The input matrix to be transposed.
     * @return The transposed matrix.
     * @throws std::invalid_argument if the input matrix is empty.
     */
    std::vector<std::vector<double>> transpose(const std::vector<std::vector<double>>& matrix);

    /**
     * Transposes a given vector.
     * @param vector The input vector to be transposed.
     * @return The transposed vector.
     * @throws std::invalid_argument if the input vector is empty.
     */
    std::vector<double> transpose(const std::vector<double>& vector);

    /**
     * sums the elements of a matrix along a specified axis.
     * @param matrix The input matrix.
     * @param axis The axis along which to sum (0 for rows, 1 for columns).
     * @return The resulting matrix after summation.
     * @throws std::invalid_argument if the axis is not 0 or 1.
     */
    std::vector<double> sum(const std::vector<std::vector<double>>& matrix, int axis);

    /**
     * Performs element-wise operation on a matrix and a scalar value.
     * @param matrix The input matrix.
     * @param scalar The scalar value to be added to each element of the matrix.
     * @param operation The operation to be performed 
     * @return The resulting matrix after the operation.
     * @throws std::invalid_argument if the operation is not one of '+', '-', '*', or '/'.
     */
    std::vector<std::vector<double>> elementwise_operation(const std::vector<std::vector<double>>& matrix, double scalar, char operation);

    /**
     * Performs element-wise operation on a vector and a scalar value.
     * @param vector The input vector.
     * @param scalar The scalar value to be added to each element of the vector.
     * @param operation The operation to be performed   
     * @return The resulting vector after the operation.
     * @throws std::invalid_argument if the operation is not one of '+', '-', '*', or '/'.
     */
    std::vector<double> elementwise_operation(const std::vector<double>& vector, double scalar, char operation);
    

    /**
     * Performs element-wise operation on two matrices.
     * @param a The first matrix.
     * @param b The second matrix.
     * @param operation The operation to be performed 
     * @return The resulting matrix after the operation.
     * @throws std::invalid_argument if the dimensions of the matrices do not match or if the operation is not one of '+', '-', '*', or '/'.
     */
    std::vector<std::vector<double>> elementwise_operation(const std::vector<std::vector<double>>& a, const std::vector<std::vector<double>>& b, char operation);

    /**
     * Performs element-wise operation on two vectors.
     * @param a The first vector.
     * @param b The second vector.
     * @param operation The operation to be performed  
     * @return The resulting vector after the operation.
     * @throws std::invalid_argument if the dimensions of the vectors do not match or if the operation is not one of '+', '-', '*', or '/'.
     */
    std::vector<double> elementwise_operation(const std::vector<double>& a, const std::vector<double>& b, char operation);

    /**
     * performs an element wise log
     * @param matrix The input matrix.
     * @return The resulting matrix after the log operation.
     */
    std::vector<std::vector<double>> log(const std::vector<std::vector<double>>& matrix);

    /**
     * Mean squared error loss function.
     * @param predictions The predicted values.
     * @param targets The target values.
     * @return The mean squared error between predictions and targets.
     */
    double mean_squared_error(const std::vector<std::vector<double>>& predictions, const std::vector<std::vector<double>>& targets);

    /**
     * Derivative of the mean squared error loss function.
     * @param predictions The predicted values. 
     * @param targets The target values.
     * @return The derivative of the mean squared error between predictions and targets.
     */
    std::vector<std::vector<double>> mean_squared_error_derivative(const std::vector<std::vector<double>>& predictions, const std::vector<std::vector<double>>& targets);

    /**
     * Binary cross-entropy loss function.
     * @param predictions The predicted probabilities.
     * @param targets The target values (0 or 1).
     * @return The binary cross-entropy loss between predictions and targets.
     */
    double binary_cross_entropy(const std::vector<std::vector<double>>& predictions, const std::vector<std::vector<double>>& targets);

    /**
     * Derivative of the binary cross-entropy loss function.
     * @param predictions The predicted probabilities.
     * @param targets The target values (0 or 1).
     * @return The derivative of the binary cross-entropy loss between predictions and targets.
     */
    std::vector<std::vector<double>> binary_cross_entropy_derivative(const std::vector<std::vector<double>>& predictions, const std::vector<std::vector<double>>& targets);

    /**
     * Categorical cross-entropy loss function.
     * @param predictions The predicted probabilities for each class.
     * @param targets The target values (one-hot encoded).
     * @return The categorical cross-entropy loss between predictions and targets.
     * 
     */
    double categorical_cross_entropy(const std::vector<std::vector<double>>& predictions, const std::vector<std::vector<double>>& targets);

    /**
     * Derivative of the categorical cross-entropy loss function.
     * @param predictions The predicted probabilities for each class.
     * @param targets The target values (one-hot encoded).
     * @return The derivative of the categorical cross-entropy loss between predictions and targets.
     */
    std::vector<std::vector<double>> categorical_cross_entropy_derivative(const std::vector<std::vector<double>>& predictions, const std::vector<std::vector<double>>& targets);

    /**
     * Calculates the mean of a matrix along a specified axis.
     * @param matrix The input matrix.
     * @param axis The axis along which to calculate the mean (0 for rows, 1 for columns).
     * @return The resulting vector after calculating the mean.
     * @throws std::invalid_argument if the axis is not 0 or 1.
     */
    std::vector<double> mean(const std::vector<std::vector<double>>& matrix, int axis);

    /**
     * calculates the standard deviation of a matrix along a specified axis.
     * @param matrix The input matrix.
     * @param axis The axis along which to calculate the standard deviation (0 for rows, 1 for columns).
     * @return The resulting vector after calculating the standard deviation.
     * @throws std::invalid_argument if the axis is not 0 or 1.
     */
    std::vector<double> std(const std::vector<std::vector<double>>& matrix, int axis);


}
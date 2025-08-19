#include "Toolkit.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK(x) do { cudaError_t err = x; if(err != cudaSuccess) { \
    std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
    exit(EXIT_FAILURE); \
} } while(0)

#define CUBLAS_CHECK(x) do { cublasStatus_t status = x; if(status != CUBLAS_STATUS_SUCCESS) { \
    std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ << std::endl; \
    exit(EXIT_FAILURE); \
} } while(0)

// transforms a 2D matrix in col-major order to a 1D vector
static std::vector<double> flatten_col_major(const std::vector<std::vector<double>>& matrix) 
{
    if (matrix.empty()) 
    {
        throw std::invalid_argument("Input matrix cannot be empty.");
    }

    int rows = matrix.size();
    int cols = matrix[0].size();
    std::vector<double> flat(rows * cols);

    for (int j = 0; j < cols; ++j) 
    {
        for (int i = 0; i < rows; ++i) 
        {
            flat[j * rows + i] = matrix[i][j];
        }
    }
    return flat;
}

// transforms a col-major 1D vector to a 2D row-major matrix
static std::vector<std::vector<double>> reshape_to_2d(const std::vector<double>& flat, int rows, int cols) 
{
    std::vector<std::vector<double>> matrix(rows, std::vector<double>(cols, 0.0));
    for (int j = 0; j < cols; ++j)
        for (int i = 0; i < rows; ++i)
            matrix[i][j] = flat[j * rows + i]; 
    return matrix;
}

namespace tk {

// --- Persistent GPU buffers ---
static double* dA = nullptr;
static double* dB = nullptr;
static double* dC = nullptr;
static size_t dA_capacity = 0;
static size_t dB_capacity = 0;
static size_t dC_capacity = 0;

// --- Persistent cuBLAS handle ---
static cublasHandle_t handle;
static bool handle_initialized = false;

static void ensure_capacity(double** d_ptr, size_t& capacity, size_t needed) {
    if (needed > capacity) {
        if (*d_ptr != nullptr) {
            CUDA_CHECK(cudaFree(*d_ptr));
        }
        CUDA_CHECK(cudaMalloc((void**)d_ptr, needed * sizeof(double)));
        capacity = needed;
    }
}

std::vector<std::vector<double>> matmul(std::vector<std::vector<double>> a, std::vector<std::vector<double>> b) 
{

    if (a.empty() || b.empty()) {
        throw std::invalid_argument("Input matrices cannot be empty.");
    }

    const int a_rows = (int)a.size();
    const int a_cols = (int)a[0].size();
    const int b_rows = (int)b.size();
    const int b_cols = (int)b[0].size();

    if (a_cols != b_rows) {
        throw std::invalid_argument("Number of columns in the first matrix must match the number of rows in the second matrix.");
    }

    std::vector<double> a_flat = flatten_col_major(a);
    std::vector<double> b_flat = flatten_col_major(b);
    std::vector<double> c_flat(a_rows * b_cols, 0.0);

    if (!handle_initialized) {
        CUBLAS_CHECK(cublasCreate(&handle));
        handle_initialized = true;
    }

    ensure_capacity(&dA, dA_capacity, a_flat.size());
    ensure_capacity(&dB, dB_capacity, b_flat.size());
    ensure_capacity(&dC, dC_capacity, c_flat.size());

    CUDA_CHECK(cudaMemcpy(dA, a_flat.data(), a_flat.size() * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, b_flat.data(), b_flat.size() * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dC, 0, c_flat.size() * sizeof(double)));

    const double alpha = 1.0;
    const double beta = 0.0;
    const int m = a_rows;
    const int k = a_cols;
    const int n = b_cols;

    CUBLAS_CHECK(
        cublasDgemm(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            n, m, k,
            &alpha,
            dB, n,
            dA, k,
            &beta,
            dC, n
        )
    );

    CUDA_CHECK(cudaMemcpy(c_flat.data(), dC, c_flat.size() * sizeof(double), cudaMemcpyDeviceToHost));

    std::vector<std::vector<double>> result = reshape_to_2d(c_flat, a_rows, b_cols);


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



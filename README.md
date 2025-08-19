# Multi-Layer Perceptron (MLP) from Scratch - C++

This project is a full implementation of a **Multi-Layer Perceptron (MLP)** from scratch using **C++**. The main goal is to build the entire infrastructure of a multilayer neural network **without relying on external machine learning libraries**, focusing on understanding and low-level implementation.

---

## 📍 Key Features

- Flexible architecture definition (number of layers and neurons per layer)
- Custom implementation of matrix operations (multiplication, transposition, etc.)
- Training with different activation functions and loss functions
- Accuracy evaluation on a test set
- CSV file loading and preprocessing for datasets

---

## 📂 Project Structure

- `MultiLayerPerceptron.*` → MLP class structure and training logic  
- `Layer.*` → Implementation of individual network layers  
- `Toolkit.*` → Mathematical utilities (matrix operations, activation functions)  
- `main.cpp` → Entry point with training/testing routines  

---

## Parallelization
- **Serial version**: Basic implementation without parallelization.
- **Parallel version (OpenMP)**: Utilizes OpenMP for parallel execution of matrix operations.
- **Parallel version (Cuda)**: Leverages CUDA for GPU acceleration of matrix operations.
- **Conclusion**: The CUDA version is, slower than the OpenMP version on this project, because the overhead of transferring data to and from the GPU outweighs the benefits of parallel execution for this specific workload, since the only parallelized file is Tollkit. If a better speedup is desired, the entire MLP should be parallelized, not just the Toolkit. Also, the architeture of the MLP interfers in what parallelization is more efficient, so the results may vary depending on the architecture used.

---
## Architecture
The MLP architecture is defined in the `main.cpp` file, where you can specify the number of layers and neurons per layer. The network is designed to handle various input sizes and can be easily modified for different datasets.


## ⚙️ Build Instructions

**Serial version:**
```bash
make
```

**Parallel version (OpenMP):**
```bash
make omp
```

**Parallel version (Cuda):**
```bash
make cuda
```

---

## 🧪 Running the Program

After compilation, run the executable:
```bash
./mlp
```

You can configure network architecture and training parameters inside the `main.cpp` file.

---

## 📊 Dataset

The model is tested on the **UCI Letter Recognition Dataset**, available at:  
🔗 [UCI Letter Recognition Dataset](https://archive.ics.uci.edu/ml/datasets/letter+recognition)

---

## 📦 Requirements

- **C++17** or higher
- **GCC** or **Clang** compiler with C++17 support
- **OpenMP** (optional, for parallel version)
- `make` utility (for compilation)
---

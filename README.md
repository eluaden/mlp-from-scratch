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

## ⚙️ Build Instructions

**Serial version:**
```bash
make
```

**Parallel version (OpenMP):**
```bash
make omp
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

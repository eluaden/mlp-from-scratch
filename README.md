# Multi-Layer Perceptron (MLP) from Scratch - C++

Este projeto é uma implementação de um **Multi-Layer Perceptron (MLP)** do zero, utilizando a linguagem **C++**. O foco principal foi construir toda a infraestrutura de uma rede neural multicamada sem utilizar bibliotecas externas de machine learning.

---

## 📍 Funcionalidades principais:

- Definição flexível de arquiteturas de rede (quantidade de camadas e neurônios por camada)
- Implementação própria de operações de matrizes (matmul, transpose, etc)
- Treinamento com diferentes funções de ativação e funções de perda
- Cálculo de acurácia sobre o conjunto de teste
- Leitura de datasets a partir de arquivos CSV

---

## 📂 Estrutura básica do código:

- `MultiLayerPerceptron.*` → Estrutura geral da rede
- `Layer.*` → Definição das camadas
- `Toolkit.*` → Operações matemáticas auxiliares
- `main.cpp` → Arquivo principal para execução dos testes

---

## ⚙️ Como compilar:

**Versão padrão (serial):**
```bash
make
```

**Versão OpenMP (paralela):**
```bash
make omp
```

## 📝 Como executar:
Após compilar, você pode executar o programa com o seguinte comando:
```bash
./mlp 
```

## Dataset utilizado:
O dataset utilizado para os testes é o **UCI letter recognition dataset**,
link: [UCI Letter Recognition Dataset](https://archive.ics.uci.edu/ml/datasets/letter+recognition)


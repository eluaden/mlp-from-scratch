#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <omp.h>
#include "MultiLayerPerceptron.h"


void load_letter_data(const std::string& path,
    std::vector<std::vector<double>>& X,
    std::vector<std::vector<double>>& y) {

    std::ifstream in(path);
    if (!in.is_open()) {
        std::cerr << "Erro ao abrir: " << path << std::endl;
        exit(1);
    }
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string token;

        std::getline(ss, token, ',');
        char letter = token[0];
        int class_idx = letter - 'A';
        std::vector<double> onehot(26, 0.0);
        onehot[class_idx] = 1.0;

        std::vector<double> feats;
        while (std::getline(ss, token, ',')) {
            feats.push_back(std::stod(token));
        }
        if (feats.size() != 16) continue;

        X.push_back(feats);
        y.push_back(onehot);
    }
}

void split_data(const std::vector<std::vector<double>>& X_all,
                const std::vector<std::vector<double>>& y_all,
                std::vector<std::vector<double>>& X_train,
                std::vector<std::vector<double>>& y_train,
                std::vector<std::vector<double>>& X_test,
                std::vector<std::vector<double>>& y_test) {
    const size_t n_train = 16000;
    for (size_t i = 0; i < X_all.size(); ++i) {
        if (i < n_train) {
            X_train.push_back(X_all[i]);
            y_train.push_back(y_all[i]);
        } else {
            X_test.push_back(X_all[i]);
            y_test.push_back(y_all[i]);
        }
    }
}

int main() {
    std::vector<std::vector<double>> X_all, y_all;
    load_letter_data("data/letter-recognition.data", X_all, y_all);
    std::vector<std::vector<double>> X_train, y_train, X_test, y_test;
    split_data(X_all, y_all, X_train, y_train, X_test, y_test);

    MultiLayerPerceptron mlp({
        Layer(256, ActivationFunction::RELU),
        Layer(128, ActivationFunction::RELU),
        Layer(64, ActivationFunction::RELU),
        Layer(32, ActivationFunction::RELU),
        Layer(26, ActivationFunction::SOFTMAX)
    });
    mlp.compile(LossFunction::CCE);

    //training time0
    double t = omp_get_wtime();

    mlp.fit(X_train, y_train, 100, 0.01, 512);

    t = omp_get_wtime() - t;
    std::cout << "Tempo de treinamento: " << t << " segundos\n";

    double acc = mlp.accuracy(X_test, y_test);
    std::cout << "Acurácia final (teste): " << acc * 100 << "%\n";
    return 0;
}
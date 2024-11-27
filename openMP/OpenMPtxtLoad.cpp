#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <chrono>
#include <omp.h>
#include <iomanip>

using namespace std;

// Funkcja do wczytania krawędzi grafu z pliku
void loadEdges(const string& filename, vector<int>& row_idx, vector<int>& col_idx) {
    ifstream file(filename);
    int row, col;
    while (file >> row >> col) {
        row_idx.push_back(row);
        col_idx.push_back(col);
    }
}

// Funkcja do wczytania cech wierzchołków z pliku
void loadFeatures(const string& filename, vector<vector<double>>& features) {
    ifstream file(filename);
    string line;
    while (getline(file, line)) {
        stringstream ss(line);
        vector<double> feature;
        double val;
        while (ss >> val) {
            feature.push_back(val);
        }
        features.push_back(feature);
    }
}

// Funkcja do mnożenia macierzy rzadkiej (CSR) z gęstą
void spmm(const vector<int>& row_idx, const vector<int>& col_idx, const vector<double>& values,
          const vector<vector<double>>& B, vector<vector<double>>& C, int num_rows, int num_cols) {
    #pragma omp parallel for
    for (int i = 0; i < num_rows; ++i) {
        for (size_t j = row_idx[i]; j < row_idx[i + 1]; ++j) {
            int col = col_idx[j];
            double val = values[j];
            for (int k = 0; k < num_cols; ++k) {
                #pragma omp atomic
                C[i][k] += val * B[col][k];
            }
        }
    }
}

int main() {
    // Załaduj dane z plików
    vector<int> row_idx, col_idx;
    loadEdges("edges.txt", row_idx, col_idx);

    // Przekształć indeksy wierszy na CSR
    int num_nodes = 2708;  // Liczba wierzchołków w Cora
    vector<int> row_ptr(num_nodes + 1, 0);
    vector<double> values(row_idx.size(), 1.0);  // Zwykle wartości w macierzy grafu są 1 w przypadku grafów nieskierowanych

    // Zaktualizuj row_ptr
    for (size_t i = 0; i < row_idx.size(); ++i) {
        row_ptr[row_idx[i] + 1]++;
    }

    for (int i = 0; i < num_nodes; ++i) {
        row_ptr[i + 1] += row_ptr[i];
    }

    // Załaduj cechy wierzchołków
    vector<vector<double>> features;
    loadFeatures("features.txt", features);
    int num_features = features[0].size();  // Liczba cech dla każdego wierzchołka

    // Stwórz gęstą macierz B
    vector<vector<double>> B(num_nodes, vector<double>(num_features, 1.0));  // Tutaj zakładamy, że B to macierz 1

    // Przygotowanie macierzy wynikowej
    vector<vector<double>> C(num_nodes, vector<double>(num_features, 0.0));

    // Pomiar czasu wykonania SpMM
    auto start = chrono::high_resolution_clock::now();
    spmm(row_ptr, col_idx, values, B, C, num_nodes, num_features);
    auto end = chrono::high_resolution_clock::now();

    // Czas wykonania
    chrono::duration<double> elapsed = end - start;
    cout << fixed << setprecision(6);
    cout << "Czas wykonania SpMM: " << elapsed.count() << " sekund" << endl;

    // // Wyświetlenie wyników
    // cout << "Pierwsze 5 wierszy macierzy C:" << endl;
    // for (int i = 0; i < 5; ++i) {
    //     for (int j = 0; j < num_features; ++j) {
    //         cout << C[i][j] << " ";
    //     }
    //     cout << endl;
    // }

    return 0;
}

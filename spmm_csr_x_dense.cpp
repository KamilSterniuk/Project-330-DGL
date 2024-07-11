#include <iostream>
#include <vector>
#include <unordered_map>

// Struktura CSR
struct CSRMatrix {
    std::vector<int> row_ptr;
    std::vector<int> col_idx;
    std::vector<double> values;
    int rows;
    int cols;
};

// Funkcja do wykonywania SpMM
std::vector<std::vector<double>> spmm(const CSRMatrix& A, const std::vector<std::vector<double>>& B) {
    // Sprawdzenie wymiarów
    if (A.cols != B.size()) {
        throw std::invalid_argument("Dimensions of matrices are not compatible for multiplication.");
    }

    int rows = A.rows;
    int cols = B[0].size();
    std::vector<std::vector<double>> C(rows, std::vector<double>(cols, 0.0));

    // Mnożenie
    for (int i = 0; i < rows; ++i) {
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            int a_col = A.col_idx[j];
            double a_val = A.values[j];

            for (int k = 0; k < cols; ++k) {
                C[i][k] += a_val * B[a_col][k];
            }
        }
    }

    return C;
}

// Funkcja do wyświetlania wyniku w formacie pełnej macierzy
void printDense(const std::vector<std::vector<double>>& C) {
    std::cout << "Dense matrix:" << std::endl;
    for (const auto& row : C) {
        for (double val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    // Przykładowa macierz A w formacie CSR
    CSRMatrix A = {
        {0, 2, 4, 6},
        {0, 2, 1, 2, 0, 1},
        {1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
        3,
        3
    };

    // Przykładowa gęsta macierz B
    std::vector<std::vector<double>> B = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0}
    };

    // Mnożenie A * B
    std::vector<std::vector<double>> C = spmm(A, B);

    // Wyświetlanie wyniku w formacie pełnej macierzy
    std::cout << "Matrix C in dense format:" << std::endl;
    printDense(C);

    return 0;
}
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
CSRMatrix spmm(const CSRMatrix& A, const CSRMatrix& B) {
    // Sprawdzenie wymiarów
    if (A.cols != B.rows) {
        throw std::invalid_argument("Dimensions of matrices are not compatible for multiplication.");
    }

    int rows = A.rows;
    int cols = B.cols;
    CSRMatrix C;
    C.rows = rows;
    C.cols = cols;

    // Tymczasowe struktury dla C
    std::vector<std::unordered_map<int, double>> tempC(rows);

    // Mnożenie
    for (int i = 0; i < rows; ++i) {
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            int a_col = A.col_idx[j];
            double a_val = A.values[j];

            for (int k = B.row_ptr[a_col]; k < B.row_ptr[a_col + 1]; ++k) {
                int b_col = B.col_idx[k];
                double b_val = B.values[k];

                tempC[i][b_col] += a_val * b_val;
            }
        }
    }

    // Przekształcenie wyniku do formatu CSR
    C.row_ptr.push_back(0);
    for (int i = 0; i < rows; ++i) {
        for (const auto& entry : tempC[i]) {
            C.col_idx.push_back(entry.first);
            C.values.push_back(entry.second);
        }
        C.row_ptr.push_back(C.col_idx.size());
    }

    return C;
}

int main() {
    // Przykładowe macierze A i B w formacie CSR
    CSRMatrix A = {
        {0, 2, 4},
        {0, 1, 0, 1},
        {1.0, 2.0, 3.0, 4.0},
        2,
        2
    };

    CSRMatrix B = {
        {0, 1, 2},
        {0, 1},
        {5.0, 6.0},
        2,
        2
    };

    // Mnożenie A * B
    CSRMatrix C = spmm(A, B);

    // Wyświetlanie wyniku
    std::cout << "C.row_ptr: ";
    for (int x : C.row_ptr) std::cout << x << " ";
    std::cout << std::endl;

    std::cout << "C.col_idx: ";
    for (int x : C.col_idx) std::cout << x << " ";
    std::cout << std::endl;

    std::cout << "C.values: ";
    for (double x : C.values) std::cout << x << " ";
    std::cout << std::endl;

    return 0;
}

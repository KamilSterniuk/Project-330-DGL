#include <torch/extension.h>
#include <vector>
#include <stdexcept>
#include <omp.h> // Dodanie nagłówka OpenMP

// Struktura CSR
struct CSRMatrix {
    std::vector<int> row_ptr; // Wskaźniki wierszy
    std::vector<int> col_idx; // Indeksy kolumn
    std::vector<double> values; // Wartości macierzy
    int rows; // Liczba wierszy
    int cols; // Liczba kolumn
};

// Funkcja do mnożenia macierzy CSR przez macierz gęstą
at::Tensor spmm_csr(const CSRMatrix& A, const at::Tensor& B) {
    // Ustawienie liczby wątków OpenMP
    omp_set_num_threads(4); // Ustawia liczbę wątków na 4 (możesz to zmienić)

    // Sprawdzenie kompatybilności wymiarów
    if (A.cols != B.size(0)) {
        throw std::invalid_argument("Dimensions of matrices are not compatible for multiplication.");
    }

    int rows = A.rows;
    int cols = B.size(1);
    at::Tensor C = torch::zeros({rows, cols}, torch::kDouble);  // Wynikowa macierz C

    // Równoległe mnożenie macierzy CSR przez macierz gęstą
    #pragma omp parallel for // Równoleglenie pętli
    for (int i = 0; i < rows; ++i) {
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
            int a_col = A.col_idx[j];
            double a_val = A.values[j];

            for (int k = 0; k < cols; ++k) {
                // Dodanie wartości do wynikowej macierzy C
                C[i][k] += a_val * B[a_col][k].item<double>();
            }
        }
    }

    return C;
}

// Rejestracja modułu PyTorch
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("spmm_csr", &spmm_csr, "CSR Sparse Matrix-Matrix Multiplication (SpMM)");
}

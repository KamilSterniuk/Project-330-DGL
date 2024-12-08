#include <torch/extension.h>
#include <vector>
#include <omp.h> // OpenMP - do równoległości

torch::Tensor spmm_csr(
    torch::Tensor indices,
    torch::Tensor indptr,
    torch::Tensor data,
    torch::Tensor dense_matrix)
{
    // Rozmiary wejściowych tensorów
    int64_t num_rows = indptr.size(0) - 1;      // Liczba węzłów
    int64_t num_heads = dense_matrix.size(1);   // Liczba głów
    int64_t feature_dim = dense_matrix.size(2); // Liczba cech (num_features)
    auto result = torch::zeros({num_rows, num_heads, feature_dim}, data.options());

// Równoległa pętla po wierszach macierzy CSR
#pragma omp parallel for
    for (int row = 0; row < num_rows; row++)
    {
        int start = indptr[row].item<int>();
        int end = indptr[row + 1].item<int>();

        for (int h = 0; h < num_heads; h++)
        { // Każda głowa działa oddzielnie
            torch::Tensor row_result = torch::zeros({feature_dim}, data.options());
            for (int i = start; i < end; i++)
            {
                int col = indices[i].item<int>();
                float edge_weight = data[i][h].item<float>(); // Waga dla konkretnej głowy
                row_result += edge_weight * dense_matrix[col][h];
            }
            result[row][h].copy_(row_result);
        }
    }

    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("spmm_csr", &spmm_csr, "CSR x Dense SpMM with 3D tensor support");
}

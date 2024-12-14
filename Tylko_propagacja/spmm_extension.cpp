#include <torch/extension.h>
#include <omp.h>

// Funkcja: spmm_csr_3d
// Mnoży macierz rzadka w formacie CSR przez macierz gęstą 3D.
// indices: [E]        - kolumny dla każdej krawędzi
// indptr: [N+1]       - wskaźniki do początków przedziałów krawędzi w wierszach
// data: [E,H]         - wagi (att) dla każdej krawędzi i head'a
// dense_matrix: [N,H,D] - cechy węzłów
//
// Wynik: [N,H,D]
//
// Dla każdego wiersza (row), heada (h) i cechy (d):
// result[row,h,d] = ∑_{edge w wierszu row} data[edge,h] * dense_matrix[col(edge),h,d]

torch::Tensor spmm_csr_3d(
    torch::Tensor indices,
    torch::Tensor indptr,
    torch::Tensor data,
    torch::Tensor dense_matrix)
{
    TORCH_CHECK(indices.dim() == 1, "indices must be 1D");
    TORCH_CHECK(indptr.dim() == 1, "indptr must be 1D");
    TORCH_CHECK(data.dim() == 2, "data must be 2D [E,H]");
    TORCH_CHECK(dense_matrix.dim() == 3, "dense_matrix must be 3D [N,H,D]");

    int64_t num_rows = indptr.size(0) - 1;
    int64_t E = indices.size(0);
    int64_t H = data.size(1);
    int64_t N = dense_matrix.size(0);
    TORCH_CHECK(dense_matrix.size(1) == H, "dense_matrix second dim must match H");
    int64_t D = dense_matrix.size(2);

    auto result = torch::zeros({num_rows, H, D}, data.options());

    auto indices_ptr = indices.data_ptr<int64_t>();
    auto indptr_ptr = indptr.data_ptr<int64_t>();
    auto data_ptr = data.data_ptr<float>();
    auto dense_ptr = dense_matrix.data_ptr<float>();
    auto result_ptr = result.data_ptr<float>();

    // indeksowanie:
    // result[row,h,d] = result_ptr[row*H*D + h*D + d]
    // dense_matrix[col,h,d] = dense_ptr[col*H*D + h*D + d]

#pragma omp parallel for collapse(2)
    for (int64_t h = 0; h < H; h++)
    {
        for (int64_t row = 0; row < num_rows; row++)
        {
            int64_t start = indptr_ptr[row];
            int64_t end = indptr_ptr[row + 1];
            float *out_row = result_ptr + row * H * D + h * D;

            for (int64_t i = start; i < end; i++)
            {
                int64_t col = indices_ptr[i];
                float edge_weight = data_ptr[i * H + h]; // data[i,h]
                const float *in_row = dense_ptr + col * H * D + h * D;

                for (int64_t d = 0; d < D; d++)
                {
                    out_row[d] += edge_weight * in_row[d];
                }
            }
        }
    }

    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("spmm_csr_3d", &spmm_csr_3d, "CSR x Dense (3D) SpMM");
}

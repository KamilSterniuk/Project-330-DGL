#include <torch/extension.h>
#include <omp.h>

// spmm_csr_h: wykonuje SPMM dla CSR x Dense per head.
// indices: [E] - kolumny
// indptr: [N+1] - wskaźniki do początków wierszy (tu wiersze to "col" z kodu PyTGeometric)
// data: [E,H] - wagi krawędzi dla każdego heada (att[e,h])
// dense_matrix: [N,H*D] - macierz cech x_proj spłaszczona po headach i D
//
// Wynik: [N,H*D]

torch::Tensor spmm_csr_h(
    torch::Tensor indices,     // int64 [E]
    torch::Tensor indptr,      // int64 [N+1]
    torch::Tensor data,        // float [E,H]
    torch::Tensor dense_matrix // float [N,H*D]
)
{
    TORCH_CHECK(indices.dim() == 1, "indices must be 1D");
    TORCH_CHECK(indptr.dim() == 1, "indptr must be 1D");
    TORCH_CHECK(data.dim() == 2, "data must be 2D [E,H]");
    TORCH_CHECK(dense_matrix.dim() == 2, "dense_matrix must be 2D [N,H*D]");

    int64_t num_rows = indptr.size(0) - 1;
    int64_t E = indices.size(0);
    int64_t H = data.size(1);
    int64_t total_feat = dense_matrix.size(1); // H*D
    TORCH_CHECK(total_feat % H == 0, "H*D must divide total_feat");
    int64_t D = total_feat / H;

    auto result = torch::zeros({num_rows, total_feat}, data.options());

    auto indices_ptr = indices.data_ptr<int64_t>();
    auto indptr_ptr = indptr.data_ptr<int64_t>();
    auto data_ptr = data.data_ptr<float>();
    auto dense_ptr = dense_matrix.data_ptr<float>();
    auto result_ptr = result.data_ptr<float>();

// Zrównoleglenie po headach i wierszach
#pragma omp parallel for collapse(2)
    for (int64_t h = 0; h < H; h++)
    {
        for (int64_t row = 0; row < num_rows; row++)
        {
            int64_t start = indptr_ptr[row];
            int64_t end = indptr_ptr[row + 1];

            float *out_row = result_ptr + row * total_feat + h * D;

            for (int64_t i = start; i < end; i++)
            {
                int64_t col = indices_ptr[i];
                float edge_weight = data_ptr[i * H + h]; // data[i,h]
                const float *in_row = dense_ptr + col * total_feat + h * D;

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
    m.def("spmm_csr_h", &spmm_csr_h, "CSR x Dense SpMM with multiple heads");
}

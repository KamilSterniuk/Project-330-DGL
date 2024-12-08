#include <torch/extension.h>
#include <omp.h>

torch::Tensor spmm_csr(
    torch::Tensor indptr, torch::Tensor indices, torch::Tensor values,
    torch::Tensor dense, int64_t N)
{
    auto result = torch::zeros({indptr.size(0) - 1, dense.size(1)}, dense.options());
    auto data_ptr = values.data_ptr<float>();
    auto indptr_ptr = indptr.data_ptr<int64_t>();
    auto indices_ptr = indices.data_ptr<int64_t>();
    auto dense_ptr = dense.data_ptr<float>();
    auto result_ptr = result.data_ptr<float>();

#pragma omp parallel for
    for (int64_t row = 0; row < indptr.size(0) - 1; row++)
    {
        for (int64_t jj = indptr_ptr[row]; jj < indptr_ptr[row + 1]; jj++)
        {
            int64_t col = indices_ptr[jj];
            float val = data_ptr[jj];
            for (int64_t k = 0; k < dense.size(1); k++)
            {
                result_ptr[row * dense.size(1) + k] += val * dense_ptr[col * dense.size(1) + k];
            }
        }
    }
    return result;
}

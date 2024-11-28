import torch
from torch import Tensor

def spmm_csr(indices: Tensor, indptr: Tensor, data: Tensor, dense_matrix: Tensor) -> Tensor:

    print("spmm plik")
    if len(dense_matrix.shape) == 3:
        batch_size, num_nodes, feature_dim = dense_matrix.shape
        dense_matrix = dense_matrix.reshape(-1, feature_dim)
    elif len(dense_matrix.shape) != 2:
        raise ValueError("Dense matrix must be 2D or 3D")
    print("spmm plik1")
    if len(indices.shape) != 1 or len(indptr.shape) != 1 or len(data.shape) != 1:
        raise ValueError("indices, indptr, and data must all be 1D tensors")
    if indptr[-1] != len(data):
        raise ValueError("The last value in indptr must match the length of the data tensor")

    print("spmm plik2")
    rows = len(indptr) - 1  # Number of rows in sparse matrix
    cols = dense_matrix.size(1)  # Number of columns in dense matrix
    
    print("spmm plik3")
    # Prepare result tensor
    result = torch.zeros((rows, cols), dtype=dense_matrix.dtype, device=dense_matrix.device)
    print("spmm plik4")
    # CSR multiplication implementation
    for row in range(rows):
        start = indptr[row]
        end = indptr[row + 1]
        for idx in range(start, end):
            col = indices[idx]
            value = data[idx]
            result[row] += value * dense_matrix[col]

    return result
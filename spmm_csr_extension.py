import torch
from torch_sparse import SparseTensor

def spmm_csr(adj: SparseTensor, x: torch.Tensor) -> torch.Tensor:
    """
    Multiplies a CSR sparse matrix (as a SparseTensor) with a dense matrix.

    Args:
        adj (SparseTensor): Sparse adjacency matrix in CSR format.
        x (torch.Tensor): Dense feature matrix of shape [num_nodes, num_features].

    Returns:
        torch.Tensor: Result of the matrix multiplication, shape [num_edges, num_features].
    """
    # Perform sparse matrix multiplication
    return adj @ x  # Equivalent to torch.sparse.mm if adj is in SparseTensor format

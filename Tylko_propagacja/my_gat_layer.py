import torch
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_max
import spmm_extension

def segment_max(src, index, num_segments):
    max_val_init = torch.full(
        (num_segments, src.size(1)),
        float('-inf'),
        device=src.device,
        dtype=src.dtype
    )
    max_val, _ = scatter_max(src, index, dim=0, out=max_val_init)
    return max_val

def segment_sum(src, index, num_segments):
    sum_val_init = torch.zeros(num_segments, src.size(1), device=src.device, dtype=src.dtype)
    sum_val = scatter_add(src, index, dim=0, out=sum_val_init)
    return sum_val

def segment_softmax(e, segment_ids, num_segments=None):
    if num_segments is None:
        num_segments = segment_ids.max().item() + 1

    max_val = segment_max(e, segment_ids, num_segments)  # [N,H]
    max_val_expanded = max_val[segment_ids]  # [E,H]

    e_exp = torch.exp(e - max_val_expanded)
    sum_val = segment_sum(e_exp, segment_ids, num_segments)  # [N,H]
    sum_val_expanded = sum_val[segment_ids]  # [E,H]

    att = e_exp / (sum_val_expanded + 1e-16)
    return att

class MyGATLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, heads=8, dropout=0.6):
        super(MyGATLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout

        self.W = torch.nn.Parameter(torch.Tensor(in_channels, heads * out_channels))
        self.a_src = torch.nn.Parameter(torch.Tensor(heads, out_channels))
        self.a_dst = torch.nn.Parameter(torch.Tensor(heads, out_channels))

        torch.nn.init.xavier_uniform_(self.W)
        torch.nn.init.xavier_uniform_(self.a_src)
        torch.nn.init.xavier_uniform_(self.a_dst)

    def forward(self, x, edge_index_or_sparse):
        N = x.size(0)

        x_proj = x @ self.W  # [N, H*D]
        x_proj = x_proj.view(N, self.heads, self.out_channels)  # [N,H,D]

        Q = x_proj
        K = x_proj

        alpha_src = (Q * self.a_src).sum(dim=2)  # [N,H]
        alpha_dst = (K * self.a_dst).sum(dim=2)  # [N,H]

        if isinstance(edge_index_or_sparse, torch.Tensor):
            # COO
            edge_index = edge_index_or_sparse
            row, col = edge_index
            e = alpha_src[row] + alpha_dst[col]  # [E,H]

            idx = torch.argsort(col)
            row = row[idx]
            col = col[idx]
            e = e[idx]

            att = segment_softmax(e, col, num_segments=N)  # [E,H]
            att_3d = att.unsqueeze(-1)  # [E,H,1]
            V_src = x_proj[row]  # [E,H,D]

            out_feat = att_3d * V_src
            out_sum = torch.zeros(N, self.heads, self.out_channels, device=x.device, dtype=x.dtype)
            out_sum = scatter_add(out_feat, col, dim=0, out=out_sum)

        else:
            # CSR (SparseTensor)
            sparse_t = edge_index_or_sparse
            row, col, _ = sparse_t.coo()

            # Sortowanie po col (tak jak w przypadku COO)
            idx = torch.argsort(col)
            row = row[idx]
            col = col[idx]

            e = alpha_src[row] + alpha_dst[col]  # [E,H]

            att = segment_softmax(e, col, num_segments=N)  # [E,H]

            # Teraz zamiast scatter_add w pythonie, używamy spmm_csr_h
            # Mamy att: [E,H], x_proj: [N,H,D], chcemy out: [N,H,D]

            # spmm_csr_h oczekuje:
            # indices: [E] kolumny CSR (tutaj używamy 'row' jako indices)
            # indptr: [N+1] definicja wierszy CSR (tu 'col' jest posortowany i służy jako row w CSR)
            # data: [E,H] (att)
            # dense_matrix: [N,H*D]

            # Budowa indptr z col (posortowanego)
            indptr = torch.zeros(N+1, dtype=torch.long, device=x.device)
            counts = torch.bincount(col, minlength=N)  # zlicza ile krawędzi do każdego wiersza (col)
            indptr[1:] = counts
            indptr = torch.cumsum(indptr, dim=0)

            x_proj_reshaped = x_proj.view(N, self.heads * self.out_channels)  # [N,H*D]

            # Wywołanie spmm
            out_sum = spmm_extension.spmm_csr_h(row, indptr, att, x_proj_reshaped)
            out_sum = out_sum.view(N, self.heads, self.out_channels)

        out = out_sum.view(N, self.heads * self.out_channels)
        out = F.dropout(out, p=self.dropout, training=self.training)
        return out

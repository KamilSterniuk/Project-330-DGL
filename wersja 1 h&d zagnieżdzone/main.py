import torch
import time
import random
import numpy as np
from torch_geometric.datasets import Planetoid
from torch_sparse import SparseTensor
from my_gat_layer import MyGATLayer

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class SimpleGATModel(torch.nn.Module):
    def __init__(self, in_channels, out_channels, heads=1):
        super(SimpleGATModel, self).__init__()
        self.gat = MyGATLayer(in_channels, out_channels, heads=heads, dropout=0.0)

    def forward(self, x, edge_index_or_sparse):
        return self.gat(x, edge_index_or_sparse)

if __name__ == "__main__":
    seed_everything(42)
    dataset = Planetoid(root='data/Planetoid', name='Cora')
    data = dataset[0]

    edge_index = data.edge_index
    num_nodes = data.num_nodes
    edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float32)

    # Tworzenie SparseTensor (CSR)
    edge_index_csr = SparseTensor(
        row=edge_index[0],
        col=edge_index[1],
        value=edge_weight,
        sparse_sizes=(num_nodes, num_nodes)
    )

    x = data.x
    y = data.y

    model = SimpleGATModel(in_channels=x.size(1), out_channels=8, heads=4)

    print("Wersja 1 - H & D zagnieżdzone.")

    model.train()
    # Propagacja z COO
    start = time.time()
    out_coo = model(x, edge_index)
    end = time.time()
    print("Propagacja COO zajęła:", (end - start)*1000, "ms")

    # Propagacja z CSR
    start = time.time()
    out_csr = model(x, edge_index_csr)
    end = time.time()
    print("Propagacja CSR zajęła:", (end - start)*1000, "ms")

    # Porównanie wyników
    are_close = torch.allclose(out_coo, out_csr, atol=1e-6)
    print("Czy wyniki COO i CSR są identyczne?", are_close)
    if not are_close:
        diff = (out_coo - out_csr).abs().max()
        print("Maksymalna różnica:", diff.item())

    model.eval()
    with torch.no_grad():
        out = model(x, edge_index)
        pred = out.max(dim=1)[1]
        test_acc = (pred[data.test_mask] == y[data.test_mask]).sum().item() / data.test_mask.sum().item()
        print("Test Accuracy:", test_acc)

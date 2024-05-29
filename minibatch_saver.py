import torch
from torch_geometric.data import DataLoader
from ogb.nodeproppred import PygNodePropPredDataset

# Ładowanie danych
dataset = PygNodePropPredDataset(name='ogbn-arxiv')
split_idx = dataset.get_idx_split()
data = dataset[0]

# Definiowanie loadera
loader = DataLoader([data], batch_size=1024, shuffle=True)

# Zapisywanie minibatchy
for i, batch in enumerate(loader):
    torch.save(batch, f'batch_{i}.pt')

print("Minibatche zostały zapisane.")
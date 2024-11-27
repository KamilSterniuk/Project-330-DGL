import torch
from torch_geometric.datasets import Planetoid
import numpy as np
import os

# Załaduj zestaw danych Planetoid
dataset_name = 'Cora'  # Możesz wybrać 'Cora', 'Citeseer' lub 'Pubmed'
dataset = Planetoid(root=os.path.join(os.getcwd(), dataset_name), name=dataset_name)

# Uzyskaj dane
data = dataset[0]
print(f"Loaded {dataset_name} dataset")

# Przygotowanie krawędzi grafu (edge_index)
edge_index = data.edge_index.numpy()

# Zapisanie krawędzi grafu do pliku
np.savetxt("edges.txt", edge_index.T, fmt="%d")

# Przygotowanie cech wierzchołków (node features) jako gęstą macierz
features = data.x.numpy()

# Zapisanie cech wierzchołków do pliku
np.savetxt("features.txt", features, fmt="%f")

print("Zapisano dane: edges.txt i features.txt")

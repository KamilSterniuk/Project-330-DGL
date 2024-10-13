import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
#from gat import GATConv2
from gat_message_and_aggregate import GATConv2
from torch_geometric.data import Data
import torch.profiler as profiler
import spmm_csr_extension 

# Define the GAT model using GATConv
class GATNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, heads):
        super(GATNet, self).__init__()
        # Define GAT layers
        self.conv1 = GATConv2(in_channels, out_channels, heads=heads, dropout=0.6)
        self.conv2 = GATConv2(out_channels * heads, out_channels, heads=1, concat=False, dropout=0.6)

    def forward(self, x, edge_index):
        # Apply the first GAT layer
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        # Apply the second GAT layer
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


dataset = Planetoid(root='/tmp/Cora', name='Cora')

data = dataset[0]


model = GATNet(in_channels=data.num_features, out_channels=16, heads=8)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

with profiler.profile(
    activities=[
        profiler.ProfilerActivity.CPU,
        profiler.ProfilerActivity.CUDA],  # If using GPU
    schedule=profiler.schedule(wait=1, warmup=1, active=3),
    on_trace_ready=profiler.tensorboard_trace_handler('./log_dir'),  # Save traces to log_dir
    record_shapes=True,
    profile_memory=True,
    with_stack=True
) as prof:
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # Step the profiler
        prof.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

model.eval()
_, pred = model(data.x, data.edge_index).max(dim=1)
correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
acc = correct / int(data.test_mask.sum())
print(f'Accuracy: {acc:.4f}')

print(prof.key_averages().table(sort_by="self_cpu_time_total"))

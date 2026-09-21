"""Multilayer graph models and the no-edge DeepSets baseline."""

from torch import nn
from torch_geometric.nn import GATv2Conv, GCNConv, SAGEConv, global_mean_pool


class StudyModel(nn.Module):
    def __init__(self, architecture, hidden=32, layers=3, input_dim=3, classes=4):
        super().__init__()
        self.architecture = architecture
        conv = {"GCN": GCNConv, "GraphSAGE": SAGEConv, "GATv2": GATv2Conv}
        self.layers = nn.ModuleList(
            [
                nn.Linear(input_dim if i == 0 else hidden, hidden)
                if architecture == "DeepSets"
                else conv[architecture](input_dim if i == 0 else hidden, hidden)
                for i in range(layers)
            ]
        )
        self.head = nn.Linear(hidden, classes)

    def representations(self, x, edges, batch):
        result = [global_mean_pool(x, batch)]
        for layer in self.layers:
            x = (layer(x) if self.architecture == "DeepSets" else layer(x, edges)).relu()
            result.append(global_mean_pool(x, batch))
        return result

    def forward(self, x, edge_index, batch):
        return self.head(self.representations(x, edge_index, batch)[-1])

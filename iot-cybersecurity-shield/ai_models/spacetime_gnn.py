"""
Spacetime Graph Neural Network for temporal attack pattern detection
"""
import torch
import torch_geometric.nn as geom_nn

class STGNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = geom_nn.GCNConv(128, 64)
        self.temporal_attn = torch.nn.MultiheadAttention(64, 4)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x, _ = self.temporal_attn(x, x, x)
        return torch.sigmoid(x)
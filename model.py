import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import SGConv


class SGC(nn.Module):
    def __init__(self, config, dataset):
        super(SGC, self).__init__()
        self.conv1 = SGConv(
            dataset.num_node_features,
            dataset.num_classes,
            K=config['model']['K'],
            cached=True
        )
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_index = add_self_loops(data.edge_index)[0]
        x = self.conv1(x, edge_index)

        return F.log_softmax(x, dim=1)
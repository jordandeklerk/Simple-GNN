import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from parser import args


class conv1(torch.nn.Module):
    def __init__(self, d_in, d_out, p):
        super(conv1, self).__init__()
        self.conv = GCNConv(d_in, d_out, cached=True)
        self.p = p

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, edge_index, mask=None):
        x = F.relu(self.conv(x, edge_index))
        x = F.dropout(x, p=self.p, training=self.training)
        return x

class conv2(torch.nn.Module):
    def __init__(self, d_in, d_out):
        super(conv2, self).__init__()
        self.conv = GCNConv(d_in, d_out, cached=True)

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x, edge_index, mask=None):
        x = self.conv(x, edge_index)
        x = F.log_softmax(x, dim=1)
        return x

class GNN(torch.nn.Module):
    def __init__(self, dataset):
        super(GNN, self).__init__()
        self.crd = conv1(dataset.num_features, args.hidden, args.dropout)
        self.cls = conv2(args.hidden, dataset.num_classes)

    def reset_parameters(self):
        self.crd.reset_parameters()
        self.cls.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.crd(x, edge_index, data.train_mask)
        x = self.cls(x, edge_index, data.train_mask)
        return x
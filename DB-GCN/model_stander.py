import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch_scatter


class GCNConv(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNConv, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x: torch.Tensor, adjacency_hat: torch.sparse_coo_tensor):
        x = self.linear(x)
        x = torch.sparse.mm(adjacency_hat, x)
        return x


class TwoLayerGCN(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_size2, output_size, dropout=0.1):
        super(TwoLayerGCN, self).__init__()

        self.conv1 = GCNConv(input_size, hidden_size)
        self.conv2 = GCNConv(hidden_size, output_size)
        self.linear = torch.nn.Linear(output_size+768, 4)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adjacency_hat: torch.sparse_coo_tensor, cls):
        x = self.dropout(x)
        x = self.conv1(x, adjacency_hat)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, adjacency_hat)

        x = torch.sigmoid(torch.sum(x, dim=0))
        clsx = torch.cat((x, cls[0]), dim=0)

        x = self.linear(clsx)

        # x = torch.sum(x, dim=0)
        x = torch.unsqueeze(x, dim=0)
        return x

#
# class graph_conv(torch.nn.Module):
#     def __init__(self, embedding_size, hidden_dim, hidden_dim2, hidden_dim3, dropout):
#         super(graph_conv, self).__init__()
#
#         self.l1 = torch.nn.Linear(2 * embedding_size, hidden_dim, bias=False)
#         self.l2 = torch.nn.Linear(hidden_dim * 2, hidden_dim, bias=True)
#         self.l3 = torch.nn.Linear(hidden_dim * 2, hidden_dim2, bias=True)
#         self.l4 = torch.nn.Linear(hidden_dim2, 4, bias=True)
#         # self.l5 = torch.nn.Linear(hidden_dim2 * 2, hidden_dim3, bias=True)
#         # self.l5 = torch.nn.Linear(hidden_dim3, 4, bias=True)
#         self.dropout = torch.nn.Dropout(dropout)
#
#     def forward(self, edges, vertices):
#         x = vertices[0]
#         edge = torch.as_tensor(edges, dtype=torch.int64)
#         x = self.l1(torch.cat([x] + [torch_scatter.scatter_add(x[edge[:, 1]], edge[:, 0], dim=0,
#                                                                dim_size=x.size(0))], dim=1))
#
#         identity = x
#         x = F.relu(self.l2(
#             torch.cat([x] + [torch_scatter.scatter_add(x[edge[:, 1]], edge[:, 0], dim=0, dim_size=x.size(0))],
#                       dim=1)))
#
#         x = x / (torch.norm(x, p=2, dim=1).unsqueeze(0).t() + 0.000001)
#         x += identity  # residual connection
#
#         x = self.dropout(F.relu(self.l3(
#             torch.cat([x] + [torch_scatter.scatter_add(x[edge[:, 1]], edge[:, 0], dim=0, dim_size=x.size(0))],
#                       dim=1))))
#         x = x / (torch.norm(x, p=2, dim=1).unsqueeze(0).t() + 0.000001)
#
#         x_target = x[:11]
#         # x = torch.squeeze(x, dim=1)
#
#         x_target = self.l4(x_target)
#
#         x_target = torch.sum(x_target, dim=0)
#
#         # x_target = F.relu(self.l4(x_target))
#         # x_target = self.l5(x_target)
#
#         x_target = torch.unsqueeze(x_target, dim=0)
#
#         return x_target


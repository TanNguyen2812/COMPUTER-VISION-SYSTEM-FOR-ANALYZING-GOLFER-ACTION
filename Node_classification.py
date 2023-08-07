import torch
import torch.nn as nn


import numpy as np

import math

def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)

def zero(x):
    """return zero."""
    return 0


def identity(x):
    """return input itself."""
    return x

def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)





def get_hop_distance(num_node, edge, max_hop=1):
    adj_mat = np.zeros((num_node, num_node))
    for i, j in edge:
        adj_mat[i, j] = 1
        adj_mat[j, i] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [
        np.linalg.matrix_power(adj_mat, d) for d in range(max_hop + 1)
    ]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(adj_matrix):
    Dl = np.sum(adj_matrix, 0)
    num_nodes = adj_matrix.shape[0]
    Dn = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    norm_matrix = np.dot(adj_matrix, Dn)
    return norm_matrix


def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A

def get_hop_distance(num_node, edge, max_hop=1):
    adj_mat = np.zeros((num_node, num_node))
    for i, j in edge:
        adj_mat[i, j] = 1
        adj_mat[j, i] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [
        np.linalg.matrix_power(adj_mat, d) for d in range(max_hop + 1)
    ]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(adj_matrix):
    Dl = np.sum(adj_matrix, 0)
    num_nodes = adj_matrix.shape[0]
    Dn = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    norm_matrix = np.dot(adj_matrix, Dn)
    return norm_matrix


def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A


class Graph:
    """The Graph to model the skeletons extracted by the openpose.
    Args:
        layout (str): must be one of the following candidates
        - openpose: 18 or 25 joints. For more information, please refer to:
            https://github.com/CMU-Perceptual-Computing-Lab/openpose#output
        - ntu-rgb+d: Is consists of 25 joints. For more information, please
            refer to https://github.com/shahroudy/NTURGB-D
        strategy (str): must be one of the follow candidates
        - uniform: Uniform Labeling
        - distance: Distance Partitioning
        - spatial: Spatial Configuration
        For more information, please refer to the section 'Partition
        Strategies' in our paper (https://arxiv.org/abs/1801.07455).
        max_hop (int): the maximal distance between two connected nodes.
            Default: 1
        dilation (int): controls the spacing between the kernel points.
            Default: 1
    """

    def __init__(self,
                 layout='coco',
                 strategy='spatial',
                 max_hop=1,
                 dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation

        assert layout in [
            'openpose-18', 'openpose-25', 'ntu-rgb+d', 'ntu_edge', 'coco'
        ]
        assert strategy in ['uniform', 'distance', 'spatial', 'agcn']
        self.get_edge(layout)
        self.hop_dis = get_hop_distance(
            self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self):
        return self.A

    def get_edge(self, layout):
        """This method returns the edge pairs of the layout."""

        if layout == 'openpose-18':
            self.num_node = 18
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(4, 3), (3, 2), (7, 6), (6, 5),
                             (13, 12), (12, 11), (10, 9), (9, 8), (11, 5),
                             (8, 2), (5, 1), (2, 1), (0, 1), (15, 0), (14, 0),
                             (17, 15), (16, 14)]
            self.edge = self_link + neighbor_link
            self.center = 1
        elif layout == 'openpose-25':
            self.num_node = 25
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(4, 3), (3, 2), (7, 6), (6, 5), (23, 22),
                             (22, 11), (24, 11), (11, 10), (10, 9), (9, 8),
                             (20, 19), (19, 14), (21, 14), (14, 13), (13, 12),
                             (12, 8), (8, 1), (5, 1), (2, 1), (0, 1), (15, 0),
                             (16, 0), (17, 15), (18, 16)]
            self.self_link = self_link
            self.neighbor_link = neighbor_link
            self.edge = self_link + neighbor_link
            self.center = 1
        elif layout == 'ntu-rgb+d':
            self.num_node = 25
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(1, 2), (2, 21), (3, 21),
                              (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21),
                              (10, 9), (11, 10), (12, 11), (13, 1), (14, 13),
                              (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                              (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.self_link = self_link
            self.neighbor_link = neighbor_link
            self.edge = self_link + neighbor_link
            self.center = 21 - 1
        elif layout == 'ntu_edge':
            self.num_node = 24
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(1, 2), (3, 2), (4, 3), (5, 2), (6, 5), (7, 6),
                              (8, 7), (9, 2), (10, 9), (11, 10), (12, 11),
                              (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                              (18, 17), (19, 18), (20, 19), (21, 22), (22, 8),
                              (23, 24), (24, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 2
        elif layout == 'coco':
            self.num_node = 17
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
                              [6, 12], [7, 13], [6, 7], [8, 6], [9, 7],
                              [10, 8], [11, 9], [2, 3], [2, 1], [3, 1], [4, 2],
                              [5, 3], [4, 6], [5, 7]]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.self_link = self_link
            self.neighbor_link = neighbor_link
            self.edge = self_link + neighbor_link
            self.center = 0
        else:
            raise ValueError(f'{layout} is not supported.')

    def get_adjacency(self, strategy):
        """This method returns the adjacency matrix according to strategy."""

        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis ==
                                                                hop]
            self.A = A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[
                                    i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.center] > self.hop_dis[
                                    i, self.center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
        elif strategy == 'agcn':
            A = []
            link_mat = edge2mat(self.self_link, self.num_node)
            In = normalize_digraph(edge2mat(self.neighbor_link, self.num_node))
            outward = [(j, i) for (i, j) in self.neighbor_link]
            Out = normalize_digraph(edge2mat(outward, self.num_node))
            A = np.stack((link_mat, In, Out))
            self.A = A
        else:

            raise ValueError('Do Not Exist This Strategy')


class unit_gcn(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 A,
                 adaptive='offset',
                 conv_pos='pre',
                 with_res=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_subsets = A.size(0)

        assert adaptive in [None, 'init', 'offset', 'importance']
        self.adaptive = adaptive
        assert conv_pos in ['pre', 'post']
        self.conv_pos = conv_pos
        self.with_res = with_res

        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(inplace=True)

        if self.adaptive == 'init':
            self.A = nn.Parameter(A.clone())
        else:
            self.register_buffer('A', A)

        if self.adaptive in ['offset', 'importance']:
            self.PA = nn.Parameter(A.clone())
            if self.adaptive == 'offset':
                nn.init.uniform_(self.PA, -1e-6, 1e-6)
            elif self.adaptive == 'importance':
                nn.init.constant_(self.PA, 1)

        if self.conv_pos == 'pre':
            self.conv = nn.Conv2d(in_channels, out_channels * A.size(0), 1)
        elif self.conv_pos == 'post':
            self.conv = nn.Conv2d(A.size(0) * in_channels, out_channels, 1)

        if self.with_res:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels))
            else:
                self.down = lambda x: x

    def forward(self, x, A=None):
        """Defines the computation performed at every call."""
        n, c, t, v = x.shape
        res = self.down(x) if self.with_res else 0

        A_switch = {None: self.A, 'init': self.A}
        if hasattr(self, 'PA'):
            A_switch.update({'offset': self.A + self.PA,
                            'importance': self.A * self.PA})
        A = A_switch[self.adaptive]

        if self.conv_pos == 'pre':
            x = self.conv(x)
            x = x.view(n, self.num_subsets, -1, t, v)
            x = torch.einsum('nkctv,kvw->nctw', (x, A)).contiguous()
        elif self.conv_pos == 'post':
            x = torch.einsum('nctv,kvw->nkctw', (x, A)).contiguous()
            x = x.view(n, -1, t, v)
            x = self.conv(x)

        return self.act(self.bn(x) + res)

    def init_weights(self):
        pass

from torch.autograd import Variable
from torch.autograd import Variable


class Node_classification(nn.Module):

    def __init__(self, in_channels=2, hidden_dim=40, num_classes=3):
        super(Node_classification, self).__init__()
        self.device = 'cuda'
        self.graph = Graph()
        A = self.graph.A
        A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        # self.register_buffer('A', A)
        self.hidden_dim = hidden_dim
        self.bn = nn.BatchNorm1d(in_channels * A.shape[1])
        self.gcn1 = unit_gcn(in_channels, hidden_dim, A)
        self.gcn2 = unit_gcn(hidden_dim, hidden_dim, A)
        self.gcn3 = unit_gcn(hidden_dim, hidden_dim, A)
        self.gcn4 = unit_gcn(hidden_dim, hidden_dim, A)
        self.relu = nn.LeakyReLU(0.01)
        self.dropout = nn.Dropout(0.5)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, num_classes)
        # self.linear2 = nn.Linear(200, 20)
        # self.linear3 = nn.Linear(20, num_classes)

    def forward(self, x):
        N, V, C = x.size()
        # x = x.permute(0,2,1)
        x = x.view(N, V * C)
        x = self.bn(x)
        x = x.view(N, V, C)
        x = x.permute(0, 2, 1)
        x = x.view(N, C, 1, V)
        x = self.gcn1(x)
        # x = self.dropout(x)
        x = self.gcn2(x)
        # x = self.dropout(x)
        x = self.gcn3(x)
        # x = self.dropout(x)
        x = self.gcn4(x)
        x = x.permute(0, 3, 2, 1)
        # x = self.dropout(x)
        x = x.view(N, V, self.hidden_dim)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)

        return x


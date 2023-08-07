import numpy as np


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
            Dn[i, i] = Dl[i] ** (-1)
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


from torch.nn.modules.batchnorm import BatchNorm2d
import math

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


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
            A_switch.update({'offset': self.A + self.PA, 'importance': self.A * self.PA})
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


class unit_tcn(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1, dilation=1, dropout=0.1):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.drop = nn.Dropout(dropout, inplace=True)
        self.stride = stride

    def forward(self, x, mask=None):
        if mask is not None:
            mask = mask.permute(0, 1, 3, 2).contiguous()
            return self.drop(self.bn(self.conv(x))) * mask
        else:
            return self.drop(self.bn(self.conv(x)))


class mstcn(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels=None,
                 ms_cfg=[(3, 1), (3, 2), (3, 3), (3, 4), ('max', 3), '1x1'],
                 stride=1):

        super().__init__()
        # Multiple branches of temporal convolution
        self.ms_cfg = ms_cfg
        num_branches = len(ms_cfg)
        self.num_branches = num_branches
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = nn.LeakyReLU(inplace=True, negative_slope=0.01)

        if mid_channels is None:
            mid_channels = out_channels // num_branches
            rem_mid_channels = out_channels - mid_channels * (num_branches - 1)
        else:
            assert isinstance(mid_channels, float) and mid_channels > 0
            mid_channels = int(out_channels * mid_channels)
            rem_mid_channels = mid_channels

        self.mid_channels = mid_channels
        self.rem_mid_channels = rem_mid_channels

        branches = []
        for i, cfg in enumerate(ms_cfg):
            branch_c = rem_mid_channels if i == 0 else mid_channels
            if cfg == '1x1':
                branches.append(nn.Conv2d(in_channels, branch_c, kernel_size=1, stride=(stride, 1)))
                continue
            assert isinstance(cfg, tuple)
            if cfg[0] == 'max':
                branches.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, branch_c, kernel_size=1), nn.BatchNorm2d(branch_c), self.act,
                        nn.MaxPool2d(kernel_size=(cfg[1], 1), stride=(stride, 1), padding=(1, 0))))
                continue
            assert isinstance(cfg[0], int) and isinstance(cfg[1], int)
            branch = nn.Sequential(
                nn.Conv2d(in_channels, branch_c, kernel_size=1), nn.BatchNorm2d(branch_c), self.act,
                unit_tcn(branch_c, branch_c, kernel_size=cfg[0], stride=stride, dilation=cfg[1]))
            branches.append(branch)

        self.branches = nn.ModuleList(branches)
        tin_channels = mid_channels * (num_branches - 1) + rem_mid_channels

        self.transform = nn.Sequential(
            nn.BatchNorm2d(tin_channels), self.act, nn.Conv2d(tin_channels, out_channels, kernel_size=1))

        self.bn = nn.BatchNorm2d(out_channels)

    def inner_forward(self, x, mask=None):
        N, C, T, V = x.shape

        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        feat = torch.cat(branch_outs, dim=1)
        feat = self.transform(feat)
        return feat

    def forward(self, x, mask=None):
        out = self.inner_forward(x, mask)
        out = self.bn(out)
        return out

    def init_weights(self):
        pass


class Channel_Att(nn.Module):
    def __init__(self, channel):
        super(Channel_Att, self).__init__()

        self.fcn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // 4, kernel_size=1),
            # nn.BatchNorm2d(channel//4),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 4, channel, kernel_size=1),
            nn.Sigmoid(),
        )
        # self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.fcn(x)


class ST_Joint_Att(nn.Module):
    def __init__(self, channel, reduct_ratio, **kwargs):
        super(ST_Joint_Att, self).__init__()

        inner_channel = channel // reduct_ratio

        self.fcn = nn.Sequential(
            nn.Conv2d(channel, inner_channel, kernel_size=1),
            nn.BatchNorm2d(inner_channel),
            nn.ReLU(),
        )
        self.conv_t = nn.Conv2d(inner_channel, channel, kernel_size=1)
        self.conv_v = nn.Conv2d(inner_channel, channel, kernel_size=1)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        N, C, T, V = x.size()
        x_t = x.mean(3, keepdims=True)
        x_v = x.mean(2, keepdims=True).transpose(2, 3)
        x_att = self.fcn(torch.cat([x_t, x_v], dim=2))
        x_t, x_v = torch.split(x_att, [T, V], dim=2)
        x_t_att = self.conv_t(x_t).sigmoid()
        x_v_att = self.conv_v(x_v.transpose(2, 3)).sigmoid()
        x_att = x_t_att * x_v_att
        return x_att


class Attention_Layer(nn.Module):
    def __init__(self, out_channel, att_type):
        super(Attention_Layer, self).__init__()

        if att_type == 'ca':
            self.att = Channel_Att(out_channel)
        elif att_type == 'st':
            self.att = ST_Joint_Att(out_channel, 4)

        self.bn = nn.BatchNorm2d(out_channel)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        res = x
        x = x * self.att(x)
        return self.act(self.bn(x) + res)

class st_gcn_unit(nn.Module):
    '''
    unit GCN-TCN
    '''

    def __init__(self, in_channels, out_channels, A, ms_cfg, stride=1, residual=True):
        super(st_gcn_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A)
        # self.tcn1 = unit_tcn(out_channels, out_channels, dilation=dilation, stride=stride)
        self.tcn1 = mstcn( out_channels, out_channels,ms_cfg=ms_cfg, stride=stride)
        # self.tcn1 = FG_TFormer(out_channels)
        self.channel_att = Channel_Att(out_channels)
        self.att = ST_Joint_Att(out_channels, 4)
        self.relu = nn.LeakyReLU()

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x, mask=None):
        y = self.gcn1(x)
        y = self.tcn1(y)
        y = y*self.channel_att(y)
        y = y*self.att(y)
        # y = y*self.channel_att(y)
        y = self.relu(y + self.residual(x))
        return y


class ST_GCN(nn.Module):
    '''
    Stacked N unit GCN-TCN ->
    '''

    def __init__(self, in_channels, n_classes, kernel_size=3):
        super(ST_GCN, self).__init__()
        self.graph = Graph()
        A = self.graph.A
        A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.data_bn = nn.BatchNorm1d(in_channels * A.shape[1])
        # self.data_bn = nn.LayerNorm(in_channels * A.shape[1])

        self.l1 = st_gcn_unit(in_channels, 64, A, ms_cfg=[(9, 1), (kernel_size, 2 ** (8 - 1)), (kernel_size, 2 ** 1)])
        self.l2 = st_gcn_unit(64, 64, A, ms_cfg=[(9, 1), (kernel_size, 2 ** (8 - 2)), (kernel_size, 2 ** 2), ])
        self.l3 = st_gcn_unit(64, 64, A, ms_cfg=[(9, 1), (kernel_size, 2 ** (8 - 3)), (kernel_size, 2 ** 3), ])
        self.l4 = st_gcn_unit(64, 64, A, ms_cfg=[(9, 1), (kernel_size, 2 ** (8 - 4)), (kernel_size, 2 ** 4), ])
        self.l5 = st_gcn_unit(64, 128, A, ms_cfg=[(9, 1), (kernel_size, 2 ** (8 - 5)), (kernel_size, 2 ** 5), ])
        self.l6 = st_gcn_unit(128, 128, A, ms_cfg=[(9, 1), (kernel_size, 2 ** (8 - 6)), (kernel_size, 2 ** 6), ])
        self.l7 = st_gcn_unit(128, 128, A, ms_cfg=[(9, 1), (kernel_size, 2 ** (8 - 7)), (kernel_size, 2 ** 7), ])
        self.l8 = st_gcn_unit(128, 128, A, ms_cfg=[(9, 1), (kernel_size, 2 ** (8 - 8)), (kernel_size, 2 ** 8), ])
        # self.l9 = st_gcn_unit(128, 128, A, ms_cfg=[(9,1), (3, 2**9), ])
        # self.l10 = st_gcn_unit(128, 128, A, ms_cfg=[(9,1), (3, 2**9), (3, 2**(8-9))])
        # self.l1 = st_gcn_unit(3, 32, A, ms_cfg=[(9,1), ])
        # self.l2 = st_gcn_unit(32,32, A, ms_cfg=[(9,1), ])
        # self.l3 = st_gcn_unit(32, 32, A, ms_cfg=[(9,1), ])
        # self.l4 = st_gcn_unit(32, 32, A, ms_cfg=[(9,1), ])
        # self.l5 = st_gcn_unit(32, 64, A, ms_cfg=[(9,1), ])
        # self.l6 = st_gcn_unit(64, 64, A, ms_cfg=[(9,1), ])
        # self.l7 = st_gcn_unit(64, 64, A, ms_cfg=[(9,1), ])
        # self.l8 = st_gcn_unit(64, 128, A, ms_cfg=[(9,1), ])
        # self.l9 = st_gcn_unit(128, 128, A, ms_cfg=[(9,1), ])

        # self.l10 = st_gcn_unit(128, 128, A, ms_cfg=[(9,1), (3, 2**10)])
        # self.l1 = st_gcn_unit(3, 64, A, ms_cfg=[(9,1)])
        # self.l2 = st_gcn_unit(64, 64, A, ms_cfg=[(9,1)])
        # self.l3 = st_gcn_unit(64, 64, A, ms_cfg=[(9,1)])
        # self.l4 = st_gcn_unit(64, 64, A, ms_cfg=[(9,1)])
        # self.l5 = st_gcn_unit(64, 128, A, ms_cfg=[(9,1)])
        # self.l6 = st_gcn_unit(128, 128, A, ms_cfg=[(9,1)])
        # self.l7 = st_gcn_unit(128, 128, A, ms_cfg=[(9,1)])
        # self.l8 = st_gcn_unit(128, 256, A, ms_cfg=[(9,1)])
        # self.l9 = st_gcn_unit(256, 256, A, ms_cfg=[(9,1)])
        # self.l10 = st_gcn_unit(256, 256, A, ms_cfg=[(9,1)])
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, n_classes)
        self.relu = nn.LeakyReLU(inplace=True)
        self.drop = nn.Dropout(p=0.5)
        # self.pooling = nn.AdaptiveMaxPool2d((None, 1))

    def forward(self, x):
        N, T, V, C = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()  # N C T V
        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        # x =  x.permute(0, 2, 1).contiguous() # N T V*C
        x = self.data_bn(x)
        # x =  x.permute(0, 2, 1).contiguous() # N T V*C
        x = x.view(N, V, C, T)
        x = x.permute(0, 2, 3, 1).contiguous()

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        # x = self.l9(x)
        # x = self.l10(x)

        # Flatten or Pooling
        # x = self.pooling(x) # N C T
        # x = x.view(N, x.size(1), T)
        #
        x = x.mean(-1)

        # C_new = x.size(1)
        # x = x.permute(0, 2, 3, 1).contiguous() # N T V C
        # x = x.view(N, T, V*C_new)
        x = x.permute(0, 2, 1).contiguous()  # N T C

        x = self.fc1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = x.permute(0, 2, 1).contiguous()
        return x



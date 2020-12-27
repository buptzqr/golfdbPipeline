import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .tgcn import ConvTemporalGraphical
from .graph import Graph
from .st_gcn_alone import st_gcn

class Model_G(nn.Module):
    r"""Spatial temporal graph convolutional networks.

    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_args (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self, in_channels, num_class, edge_importance_weighting, graph_args, **kwargs):
        super().__init__()
        # load graph
        self.graph = Graph(**graph_args)
        # A就是根据距离构成的标准化以后的邻接矩阵
        A = torch.tensor(self.graph.A, dtype=torch.float32,
                         requires_grad=False)
        self.register_buffer('A', A)

        # build networks
        # 空间kernel的size就是根据不同的策略将临近点映射成了几个subset
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            # st_gcn(64, 128, kernel_size, 1, **kwargs),
            # st_gcn(128, 128, kernel_size, 1, **kwargs),
            # st_gcn(128, 128, kernel_size, 1, **kwargs),
            # st_gcn(128, 256, kernel_size, 1, **kwargs),
            # st_gcn(256, 256, kernel_size, 1, **kwargs),
            # st_gcn(256, 256, kernel_size, 1, **kwargs),
        ))
        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        # fcn for prediction
        self.fcn = nn.Conv2d(130, num_class, kernel_size=1)  # 改
        # 这个以后得手动改13帧了
        self.lstm_layers = 1
        self.bidirectional = True
        self.lstm_hidden = 64
        self.lstm = nn.LSTM(112, self.lstm_hidden, self.lstm_layers,
                            batch_first=True, bidirectional=True)  # 改
        self.lin = nn.Linear(2 * self.lstm_hidden, 9)

    def init_hidden(self, batch_size):
        if self.bidirectional:
            return (Variable(torch.zeros(2*self.lstm_layers, batch_size, self.lstm_hidden).cuda(), requires_grad=True),
                    Variable(torch.zeros(2*self.lstm_layers, batch_size, self.lstm_hidden).cuda(), requires_grad=True))
        else:
            return (Variable(torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden).cuda(), requires_grad=True),
                    Variable(torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden).cuda(), requires_grad=True))

    def forward(self, x):

        # data normalization
        N, C, T, V, M = x.size()
        # print("n c t v m")
        # print(x.size())
        # N, C, T, V, M -> N, M, V, C, T
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        # N, M, C, T, V
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)
        ori_x = x.view(N * M, T, -1)
        # print("ori_x shape")
        # print(ori_x.shape)
        # print("gcn input")
        # print(x.shape)

        # forwad
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)
        # print("gcn output")
        # print(x.shape)
        # global pooling
        x = x.mean(dim=3)
        # print("after pool")
        # print(x.shape)
        x = x.view(N, T, -1, M).mean(dim=3)
        # print("after view")
        # print(x.shape)  # batch,timesteps,-1
        x = torch.cat((x, ori_x), 2)

        # print(x.shape)
        # prediction
        batch_size = N
        self.hidden = self.init_hidden(batch_size)
        x,  state = self.lstm(x, self.hidden)
        # print("after lstm")
        # print(x.shape)  # batch,timesteps,-1
        x = self.lin(x)
        # print("after lin")
        # print(x.shape)
        ##############使用fcn####################
        # x = x.view(batch_size * T, -1)
        # x = torch.cat((x, ori_x), 1)
        # x = x.view(batch_size * T, -1, 1, 1)
        # print("after concate")
        # print(x.shape)
        # x = self.fcn(x)
        # print("after fcn")
        # print(x.shape)
        #########################################
        x = x.view(N*T, 9)

        return x

    def extract_feature(self, x):

        # data normalization
        x = x.permute(0,2,1,3,4).contiguous()
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)
        ori_x = x.view(N * M, T, -1)
        # print("ori")
        # print(ori_x.shape)
        # forwad
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            # print("enter")
            x, _ = gcn(x, self.A * importance)
            # print(x.shape)
        # print("output x")
        # print(x.shape)
        x = x.mean(dim=3)
        # print("pool x")
        # print(x.shape)
        x = x.view(N, T, -1, M).mean(dim=3)
        # print("final x")
        # print(x.shape)
        x = torch.cat((x, ori_x), 2)
        batch_size = N
        self.hidden = self.init_hidden(batch_size)
        x,  state = self.lstm(x, self.hidden)  # batch,timesteps,-1
        # print("input x")
        # print(x.shape)

        return x



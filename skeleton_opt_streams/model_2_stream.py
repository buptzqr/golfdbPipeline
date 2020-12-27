import sys
sys.path.append('..')
from MobileNetV2 import MobileNetV2
from torch.autograd import Variable
import torch.nn as nn
import torch
from data.config import cfg
from gcn.st_gcn import Model_G


class EventDetector(nn.Module):
    def __init__(self, pretrain, width_mult, lstm_layers, lstm_hidden, bidirectional=True, dropout=True):
        super(EventDetector, self).__init__()
        self.width_mult = width_mult
        self.lstm_layers = lstm_layers
        self.lstm_hidden = lstm_hidden
        self.bidirectional = bidirectional
        self.dropout = dropout
        ######for gcn######
        self.in_channels = 3
        self.num_class = 9
        self.edge_importance_weighting = True
        self.gcn_feature = 128
        self.graph_args = {'layout': 'customer settings','strategy': 'spatial', 'max_hop': 1, 'dilation': 1}
        self.st_gcn_net = Model_G(self.in_channels, self.num_class,self.edge_importance_weighting,self.graph_args)
        net = MobileNetV2(width_mult=width_mult)
        state_dict_mobilenet = torch.load('../mobilenet_v2.pth.tar')
        state_dict_st_gcn_net = torch.load('../gcn_spotting_net_8000.pth.tar')
        if pretrain:
            net.load_state_dict(state_dict_mobilenet)
            self.st_gcn_net.load_state_dict(state_dict_st_gcn_net['model_state_dict'])
        
        self.cnn = nn.Sequential(*list(net.children())[0][:19])
        self.rnn = nn.LSTM(int(1280*width_mult if width_mult > 1.0 else 1280),
                           self.lstm_hidden, self.lstm_layers,
                           batch_first=True, bidirectional=bidirectional)
        if self.bidirectional:
            if cfg.FRAME_13_OPEN:
                self.lin = nn.Linear(2*self.lstm_hidden+self.gcn_feature, 14)
            else:
                self.lin = nn.Linear(2*self.lstm_hidden+self.gcn_feature, 9)
        else:
            if cfg.FRAME_13_OPEN:
                self.lin = nn.Linear(self.lstm_hidden+self.gcn_feature, 14)
            else:
                self.lin = nn.Linear(self.lstm_hidden+self.gcn_feature, 9)
        if self.dropout:
            self.drop = nn.Dropout(0.5)

    def init_hidden(self, batch_size):
        if self.bidirectional:
            return (Variable(torch.zeros(2*self.lstm_layers, batch_size, self.lstm_hidden).cuda(), requires_grad=True),
                    Variable(torch.zeros(2*self.lstm_layers, batch_size, self.lstm_hidden).cuda(), requires_grad=True))
        else:
            return (Variable(torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden).cuda(), requires_grad=True),
                    Variable(torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden).cuda(), requires_grad=True))

    def forward(self, x, k,lengths=None):
        k = self.st_gcn_net.extract_feature(k)
        batch_size, timesteps, C, H, W = x.size()
        self.hidden = self.init_hidden(batch_size)

        # CNN forward
        c_in = x.view(batch_size * timesteps, C, H, W)
        c_out = self.cnn(c_in)
        c_out = c_out.mean(3).mean(2)
        if self.dropout:
            c_out = self.drop(c_out)

        # LSTM forward
        r_in = c_out.view(batch_size, timesteps, -1)
        r_out, states = self.rnn(r_in, self.hidden)
        # print("after lstm")
        # print(r_out.shape)
        # print(k.shape)
        concat_out = torch.cat((r_out, k), 2)
        out = self.lin(concat_out)
        if cfg.FRAME_13_OPEN:
            out = out.view(batch_size*timesteps, 14)
        else:
            out = out.view(batch_size*timesteps, 9)

        return out

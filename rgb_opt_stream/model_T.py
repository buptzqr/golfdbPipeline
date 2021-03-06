import sys
sys.path.append('..')
import torch
import torch.nn as nn
from torch.autograd import Variable
from MobileNetV2 import MobileNetV2
from data.config import cfg


class EventDetector_OPT(nn.Module):
    def __init__(self, pretrain, width_mult, lstm_layers, lstm_hidden, bidirectional=True, dropout=True):
        super(EventDetector_OPT, self).__init__()
        self.width_mult = width_mult
        self.lstm_layers = lstm_layers
        self.lstm_hidden = lstm_hidden
        self.bidirectional = bidirectional
        self.dropout = dropout

        net = MobileNetV2(width_mult=width_mult)
        state_dict_mobilenet = torch.load('/home/zqr/codes/MyGolfDB/mobilenet_v2.pth.tar')
        if pretrain:
            net.load_state_dict(state_dict_mobilenet)

        self.cnn = nn.Sequential(*list(net.children())[0][:19])
        self.rnn = nn.LSTM(int(1280*width_mult if width_mult > 1.0 else 1280),
                           self.lstm_hidden, self.lstm_layers,
                           batch_first=True, bidirectional=bidirectional)
        if self.bidirectional:
            if cfg.FRAME_13_OPEN:
                self.lin = nn.Linear(2*self.lstm_hidden, 14)
            else:
                self.lin = nn.Linear(2*self.lstm_hidden, 9)
        else:
            if cfg.FRAME_13_OPEN:
                self.lin = nn.Linear(self.lstm_hidden, 14)
            else:
                self.lin = nn.Linear(self.lstm_hidden, 9)
        if self.dropout:
            self.drop = nn.Dropout(0.5)

    def init_hidden(self, batch_size):
        if self.bidirectional:
            return (Variable(torch.zeros(2*self.lstm_layers, batch_size, self.lstm_hidden).cuda(), requires_grad=True),
                    Variable(torch.zeros(2*self.lstm_layers, batch_size, self.lstm_hidden).cuda(), requires_grad=True))
        else:
            return (Variable(torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden).cuda(), requires_grad=True),
                    Variable(torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden).cuda(), requires_grad=True))

    def forward(self, x, lengths=None):
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
        out = self.lin(r_out)
        if cfg.FRAME_13_OPEN:
            out = out.view(batch_size*timesteps, 14)
        else:
            out = out.view(batch_size*timesteps, 9)

        return out

    def extract_feature(self,x,lengths=None):
        batch_size, timesteps, C, H, W = x.size()
        self.hidden = self.init_hidden(batch_size)

        # CNN forward
        c_in = x.view(batch_size * timesteps, C, H, W)
        c_out = self.cnn(c_in)
        c_out = c_out.mean(3).mean(2)
        # 如果是在lstm之前融合，我直接把dropout放在concat之后了
        # if self.dropout:
        #     c_out = self.drop(c_out)
        
        # LSTM forward
        r_in = c_out.view(batch_size, timesteps, -1)
        # 在lstm之前融合
        r_out = r_in
        # 在lstm之后融合
        # r_out, states = self.rnn(r_in, self.hidden)
        
        return r_out
        
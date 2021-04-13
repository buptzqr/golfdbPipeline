import sys
sys.path.append('..')
import torch
import torch.nn as nn
from torch.autograd import Variable
from data.config import cfg
from model_T import EventDetector_OPT
from model_rgb import EventDetector_RGB

class EventDetector(nn.Module):
    def __init__(self, pretrain, width_mult, lstm_layers, lstm_hidden, bidirectional=True, dropout=True):
        super(EventDetector, self).__init__()
        self.width_mult = width_mult
        self.lstm_layers = lstm_layers
        self.lstm_hidden = lstm_hidden
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.rgb_net = EventDetector_RGB(pretrain,width_mult,lstm_layers,lstm_hidden,bidirectional,dropout)
        self.opt_net = EventDetector_OPT(pretrain,width_mult,lstm_layers,lstm_hidden,bidirectional,dropout)
        from collections import OrderedDict
        state_dict_opt = torch.load('./swingnet_1700.pth.tar')
        state_dict_rgb = torch.load('./swingnet_1800.pth.tar')
        
        self.opt_net.load_state_dict(state_dict_opt['model_state_dict'])
        self.rgb_net.load_state_dict(state_dict_rgb['model_state_dict'])
        self.rnn = nn.LSTM(int(2560*width_mult if width_mult > 1.0 else 2560),
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

    def forward(self, opt_img, rgb_img, lengths=None):
        batch_size, timesteps, C, H, W = rgb_img.size()
        self.hidden = self.init_hidden(batch_size)

        opt_img = self.opt_net.extract_feature(opt_img)
        rgb_img = self.rgb_net.extract_feature(rgb_img)
        
        r_in = torch.cat((rgb_img, opt_img), 2)
        
        if self.dropout:
            r_in = self.drop(r_in)
        r_out, states = self.rnn(r_in, self.hidden)

        out = self.lin(r_out)
        if cfg.FRAME_13_OPEN:
            out = out.view(batch_size*timesteps, 14)
        else:
            out = out.view(batch_size*timesteps, 9)

        return out


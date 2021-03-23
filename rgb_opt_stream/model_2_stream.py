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
        # state_dict_opt = torch.load('./opt_swingnet_700.pth.tar')
        # state_dict_rgb = torch.load('./rgb_swingnet_1800.pth.tar')
        # self.rgb_net.load_state_dict(state_dict_rgb)
        # self.opt_net.load_state_dict(state_dict_opt)
        
        if self.bidirectional:
            if cfg.FRAME_13_OPEN:
                self.lin = nn.Linear(2*2*self.lstm_hidden, 14)
            else:
                self.lin = nn.Linear(2*2*self.lstm_hidden, 9)
        else:
            if cfg.FRAME_13_OPEN:
                self.lin = nn.Linear(2*self.lstm_hidden, 14)
            else:
                self.lin = nn.Linear(2*self.lstm_hidden, 9)
        if self.dropout:
            self.drop = nn.Dropout(0.5)

    def forward(self, opt_img, rgb_img, lengths=None):
        batch_size, timesteps, C, H, W = rgb_img.size()
        opt_img = self.opt_net.extract_feature(opt_img)
        rgb_img = self.rgb_net.extract_feature(rgb_img)
        r_out = torch.cat((rgb_img, opt_img), 2)
        out = self.lin(r_out)
        if cfg.FRAME_13_OPEN:
            out = out.view(batch_size*timesteps, 14)
        else:
            out = out.view(batch_size*timesteps, 9)

        return out


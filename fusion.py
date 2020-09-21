from model import EventDetector
from dataloader_T import GolfDB_T, Normalize_T
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataloader import GolfDB, ToTensor, Normalize
import torch.nn.functional as F
import numpy as np
from util import correct_preds
import matplotlib.pyplot as plt 
from myeval import myeval
import sys

rgb_count = 0
def fusion_correct_preds(rgb_probs, optical_probs, tol, events):
    global rgb_count
    preds = np.zeros(len(events))
    for i in range(8):
        rgb_preds = np.argsort(rgb_probs[:, i])[-1]
        optical_preds = np.argsort(optical_probs[:, i])[-1]
        if rgb_preds > optical_preds:
            preds[i] = rgb_preds
            rgb_count += 1
        else:
            preds[i] = optical_preds
    deltas = np.abs(events-preds)
    correct = (deltas <= tol).astype(np.uint8)
    return preds, deltas, correct

if __name__ == '__main__':
    split = 1
    seq_length = 64
    n_cpu = 6

    model = EventDetector(pretrain=True,
                          width_mult=1,
                          lstm_layers=1,
                          lstm_hidden=256,
                          bidirectional=True,
                          dropout=False)

    rgb_save_dict = torch.load('swingnet_1600.pth.tar')
    model.load_state_dict(rgb_save_dict['model_state_dict'])
    model.cuda()
    model.eval()

    _,_,rgb_probs,rgb_tols,rgb_events = myeval(model, split, seq_length, n_cpu, False, 1)
    
    optical_save_dict = torch.load('swingnet_1200.pth.tar')
    model.load_state_dict(optical_save_dict['model_state_dict'])
    model.cuda()
    model.eval()    
    _, _, optical_probs, optical_tols, optical_events = myeval(model, split, seq_length, n_cpu, False, 0)
    
    if len(optical_probs) != len(rgb_probs):
        print("there is error in fusion part")
        sys.exit(1)
    
    corrects = []
    for i in range(len(optical_probs)):
        _,_,c = fusion_correct_preds(rgb_probs[i], optical_probs[i], rgb_tols[i], optical_events[i])
        corrects.append(c)
    
    PCE = np.mean(corrects)
    print(rgb_count)
    print(PCE)
    
    

    
from model import EventDetector
import torch
from torch.utils.data import DataLoader
from displyDataloader import GolfDB_T, Normalize_T
from torchvision import transforms
from displyDataloader import GolfDB, ToTensor, Normalize
import torch.nn.functional as F
import numpy as np
from util import correct_preds
import matplotlib.pyplot as plt


def eval(model, split, seq_length, n_cpu, disp):
    # 非光流部分
    dataset = GolfDB(data_file='/home/zqr/codes/data/data_info.txt',
                     vid_dir='/home/zqr/codes/data/imagesFolder_160',
                     transform=None,
                     myMean=[0.485, 0.456, 0.406],
                     myStd=[0.229, 0.224, 0.225],
                     )

    # # 光流部分
    # dataset = GolfDB_T(data_file='/home/zqr/codes/data/data_info.txt',
    #                    transform=None)

    data_loader = DataLoader(dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=n_cpu,
                             drop_last=False)

    correct = []
    preds_collect = []

    for i, sample in enumerate(data_loader):
        preds = []
        images, labels = sample['images'], sample['labels']
        # full samples do not fit into GPU memory so evaluate sample in 'seq_length' batches
        batch = 0
        while batch * seq_length < images.shape[1]:
            if (batch + 1) * seq_length > images.shape[1]:
                image_batch = images[:, batch * seq_length:, :, :, :]
            else:
                image_batch = images[:, batch *
                                     seq_length:(batch + 1) * seq_length, :, :, :]
            logits = model(image_batch.cuda())
            if batch == 0:
                probs = F.softmax(logits.data, dim=1).cpu().numpy()
            else:
                probs = np.append(probs, F.softmax(
                    logits.data, dim=1).cpu().numpy(), 0)
            batch += 1
        for idx in range(9):
            # print(idx)
            preds.append(np.argsort(probs[:, idx])[-1])
        preds_collect.append(preds)
        # _, _, _, _, c = correct_preds(probs, labels.squeeze())
        # if disp:
        #     print(i, c)
        # correct.append(c)
    # PCE = np.mean(correct)
    return preds_collect


if __name__ == '__main__':

    split = 1
    seq_length = 64
    n_cpu = 6
    index = 1800

    model = EventDetector(pretrain=True,
                          width_mult=1.,
                          lstm_layers=1,
                          lstm_hidden=256,
                          bidirectional=True,
                          dropout=False)
    # save_dict = torch.load('models/swingnet_{}.pth.tar'.format(index))
    save_dict = torch.load('swingnet_1600.pth.tar')
    model.load_state_dict(save_dict['model_state_dict'])
    model.cuda()
    model.eval()
    preds = eval(model, split, seq_length, n_cpu, True)
    print(preds)

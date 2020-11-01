from model import EventDetector
import torch
from torch.utils.data import DataLoader
from displyDataloader import GolfDB_DIS_OPT, Normalize_T
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
from util import correct_preds
import matplotlib.pyplot as plt
from data.config import cfg
import os


def eval(model, split, seq_length, n_cpu, disp):

    # # 光流部分
    dataset = GolfDB_DIS_OPT()

    data_loader = DataLoader(dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=n_cpu,
                             drop_last=False)

    correct = []
    preds_collect = {}

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
        for idx in range(14):
            preds.append(np.argsort(probs[:, idx])[-1])
        preds_collect[labels[0, 0].item()] = preds
    return preds_collect


if __name__ == '__main__':

    split = cfg.SPLIT
    seq_length = cfg.SEQUENCE_LENGTH
    n_cpu = cfg.CPU_NUM

    model = EventDetector(pretrain=True,
                          width_mult=1.,
                          lstm_layers=1,
                          lstm_hidden=256,
                          bidirectional=True,
                          dropout=False)
    save_dict = torch.load(cfg.TEST_MODEL)
    model.load_state_dict(save_dict['model_state_dict'])
    model.cuda()
    model.eval()
    preds = eval(model, split, seq_length, n_cpu, True)
    print(preds)
    for k, v in preds.items():
        src_path = os.path.join(cfg.TEST_IMGS_DIR, str(k))
        dst_path = os.path.join(cfg.TEST_RESULT_PATH, str(k))
        if not os.path.exists(dst_path):
            os.mkdir(dst_path)
        for i in range(13):
            src_img = os.path.join(src_path, "{:0>4d}.jpg".format(v[i]))
            dst_img = os.path.join(dst_path, "{:0>4d}_{}.jpg".format(i, v[i]))
            os.system("cp {} {}".format(src_img, dst_img))

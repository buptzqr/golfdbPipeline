import sys
sys.path.append("..")
from dataloader_2_stream import GolfDB_2_stream
from model_2_stream import EventDetector
from util import *
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from torch.utils.tensorboard import SummaryWriter
from myeval import myeval
from data.config import cfg
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# summary = [0, 0, 0, 0, 0, 0, 0, 0]  #统计各个关键帧检测出错的数目

def myeval(model, split, seq_length, n_cpu, disp, stream_choice=0):
    videosNum = 0  # 统计验证集的视频数量
    dataset = GolfDB_2_Stream(data_file='../data/val_split_{}.pkl'.format(split),
                              vid_dir=cfg.OPT_RESIZE_FILE_PATH,
                              seq_length=seq_length,
                              train=False)
    data_loader = DataLoader(dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=n_cpu,
                             drop_last=False)

    correct = []
    # 这三个是为了做融合用
    all_probs = []
    all_tols = []
    all_events = []

    for i, sample in enumerate(data_loader):
        videosNum += 1
        # print(videosNum)
        images, opt_images, labels = sample['images'], sample['opt_images'], sample['labels']
        # full samples do not fit into GPU memory so evaluate sample in 'seq_length' batches
        batch = 0
        while batch * seq_length < images.shape[1]:
            if (batch + 1) * seq_length > images.shape[1]:
                image_batch = images[:, batch * seq_length:,:,:,:]
                opt_images_batch = opt_images[:, batch * seq_length:,:,:,:]
            else:
                image_batch = images[:, batch *
                                     seq_length: (batch + 1) * seq_length,: ,: ,: ]
                opt_images_batch = opt_images[:, batch *
                                     seq_length: (batch + 1) * seq_length,:,:,: ]
            # print(image_batch.shape)
            # print(keypoints_batch.shape)   
            logits = model(image_batch.cuda(),opt_images_batch.cuda())
            if batch == 0:
                probs = F.softmax(logits.data, dim=1).cpu().numpy()
            else:
                probs = np.append(probs, F.softmax(
                    logits.data, dim=1).cpu().numpy(), 0)
            batch += 1
        events, preds, _, tol, c = correct_preds(probs, labels.squeeze())
        all_probs.append(probs)
        all_tols.append(tol)
        all_events.append(events)
        # 统计信息
        # for i, item in enumerate(c):
        #     if c[i] == 0:
        #         summary[i] += 1
        # if c[0] ==0 or c[7]==0:
        #     info = str((preds - events).tolist())
        #     summaryFile.write(info)
        #     summaryFile.write(' ')
        #     summaryFile.write(tol)
        #     summaryFile.write('\n')
        # else:
        #     summaryFile.write('\n')
        if disp:
            # print(i, c)
            print("ground truth:")
            print(events)
            print("preds:")
            print(preds)
        correct.append(c)
    PCE = np.mean(correct)
    # summaryFile.close()
    return PCE, videosNum, all_probs, all_tols, all_events


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
    PCES = {}
    vNum = 0
    for i in range(1, 3):
        index = i*10
        print('swingnet_{}.pth.tar'.format(index))
        save_dict = torch.load('./models/swingnet_{}.pth.tar'.format(index))
        model.load_state_dict(save_dict['model_state_dict'])
        model.cuda()
        model.eval()
        PCE, vNum, _, _, _ = myeval(
            model, split, seq_length, n_cpu, False, 0)
        print("{}:{}".format(i,PCE))
        PCES[index] = PCE
    if cfg.FRAME_13_OPEN:
        print("13 frames")
        print('Average PCE: {}'.format(PCES))
        print("video file num:{}".format(vNum))
    else:
        print("8 frames")
        print('split:{}  Average PCE: {}'.format(split, PCES))
        print("video file num:{}".format(vNum))
    # print("summary:{}".format(summary))

    # # 绘图
    y_val = list(PCES.values())
    x_val = list(PCES.keys())

    plt.plot(x_val, y_val, linewidth=5)

    # 设置图表标题，并给坐标轴加上标签
    plt.title("val_precision", fontsize=24)
    plt.xlabel("iter per 100", fontsize=14)
    plt.ylabel("acc val", fontsize=14)

    # 设置刻度标记的大小
    plt.tick_params(axis='both', labelsize=14)
    if cfg.FRAME_13_OPEN:
        plt.savefig("./image/13_frames")
    else:
        plt.savefig("./image/8_frames_split{}".format(split))

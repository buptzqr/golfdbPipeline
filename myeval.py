from model import EventDetector
from dataloader_T import GolfDB_T, Normalize_T
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataloader import GolfDB, ToTensor, Normalize
from dataloader13 import GolfDB_13, ToTensor_13, Normalize_13, Normalize_T_13
import torch.nn.functional as F
import numpy as np
from util import correct_preds
import matplotlib.pyplot as plt
from data.config import cfg

# summary = [0, 0, 0, 0, 0, 0, 0, 0]  #统计各个关键帧检测出错的数目


def myeval(model, split, seq_length, n_cpu, disp, stream_choice=0):
    # summaryFile = open("summary_opt_{}.txt".format(split),"w")
    videosNum = 0  # 统计验证集的视频数量
    if cfg.FRAME_13_OPEN:
        dataset = GolfDB_13(data_file=cfg.OUR_PKL_FILE_PATH,
                            json_dir=cfg.VAL_JSON_PATH,
                            dataloader_opt=cfg.DATAOPT,
                            seq_length=64,
                            train=False)
    else:
        if stream_choice == 1:  # 默认使用光流分支
            # 评价非光流法
            dataset = GolfDB(data_file='/home/zqr/codes/GolfDB/data/val_split_{}.pkl'.format(split),
                             vid_dir=cfg.VIDEO_160_PATH,
                             seq_length=seq_length,
                             transform=None,
                             myMean=[0.485, 0.456, 0.406],
                             myStd=[0.229, 0.224, 0.225],
                             train=False)

        else:  # 评价光流法
            dataset = GolfDB_T(data_file='/home/zqr/codes/GolfDB/data/val_split_{}.pkl'.format(split),
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
            print(events)
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
                          width_mult=1,
                          lstm_layers=1,
                          lstm_hidden=256,
                          bidirectional=True,
                          dropout=False)
    PCES = {}
    vNum = 0
    for i in range(1, 6):
        index = i*100
        print('swingnet_{}.pth.tar'.format(index))
        save_dict = torch.load('./models/swingnet_{}.pth.tar'.format(index))
        model.load_state_dict(save_dict['model_state_dict'])
        model.cuda()
        model.eval()
        PCE, vNum, _, _, _ = myeval(
            model, split, seq_length, n_cpu, True, 0)
        PCES[index] = PCE

    print('split:{}  Average PCE: {}'.format(split, PCES))
    print("video file num:{}".format(vNum))
    # print("summary:{}".format(summary))

    # # 绘图
    # y_val = list(PCES.values())
    # x_val = list(PCES.keys())

    # plt.plot(x_val, y_val, linewidth=5)

    # # 设置图表标题，并给坐标轴加上标签
    # plt.title("val_precision", fontsize=24)
    # plt.xlabel("iter per 100", fontsize=14)
    # plt.ylabel("acc val", fontsize=14)

    # # 设置刻度标记的大小
    # plt.tick_params(axis='both', labelsize=14)
    # plt.savefig("./image/split{}".format(split))

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
        if stream_choice == 1:
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
            print("ground truth:")
            print(events)
            print("preds:")
            print(preds)
        correct.append(c)
    PFCR = np.mean(correct,axis=0)
    PCE = np.mean(correct)
    # summaryFile.close()
    return PCE, videosNum, all_probs, all_tols, all_events,PFCR


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
    # model = torch.nn.parallel.DataParallel(model)
    PCES = {}
    vNum = 0
    for i in range(1, 2):
        index = i*100
        print('swingnet_{}.pth.tar'.format(index))
        save_dict = torch.load('/home/zqr/data/models/optical/13/swingnet_{}.pth.tar'.format(index))
        new_state_dict = save_dict['model_state_dict']
        # from collections import OrderedDict
        # new_state_dict = OrderedDict()
        # for k, v in save_dict['model_state_dict'].items():
        #     name = k[7:] # remove `module.`
        #     new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        model.cuda()
        model.eval()
        PCE, vNum, _, _, _ ,PFCR= myeval(
            model, split, seq_length, n_cpu, True, 0)
        PCES[index] = PCE
    if cfg.FRAME_13_OPEN:
        print("13 frames")
        print('Average PCE: {}'.format(PCES))
        print("video file num:{}".format(vNum))
        print("per frame correct ratio:{}".format(PFCR))
    else:
        print("8 frames")
        print('split:{}  Average PCE: {}'.format(split, PCES))
        print("video file num:{}".format(vNum))
        print("per frame correct ratio:{}".format(PFCR))

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

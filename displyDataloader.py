import os.path as osp
import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import sys
from data.config import cfg

# 用于检测目标视频的dataloader


def Normalize_T(sample):
    images, labels = sample['images'], sample['labels']
    images = np.asarray(images)
    images = images.astype(np.float32)
    labels = np.asarray(labels)
    imgsMean = np.mean(images, axis=(1, 2))
    imgsMean = imgsMean.reshape(-1, 1, 1, 3)
    images = np.subtract(images, imgsMean)
    images = images.transpose((0, 3, 1, 2))
    images = torch.from_numpy(images)
    labels = torch.from_numpy(labels)
    return {'images': images, 'labels': labels}


class GolfDB_DIS_OPT(Dataset):
    def __init__(self, transform=None):
        self.video_dir = cfg.TEST_VIDEO_PAHT
        self.transform = transform
        self.videos_name = []  # 用来存储video名称
        for video in os.listdir(self.video_dir):
            self.videos_name.append(video.split('.')[0])
        self.opt_img_dir = cfg.TEST_OPT_DIR

    def __len__(self):
        return len(self.videos_name)

    def __getitem__(self, idx):
        images = []
        labels = []
        img_dir = os.path.join(self.opt_img_dir, self.videos_name[idx])
        img_num = 0
        for _ in os.listdir(img_dir):
            img_num += 1
        for img_name in range(0, img_num):
            # full clip
            img = cv2.imread(os.path.join(
                img_dir, "{:0>4d}.jpg".format(img_name)))
            transImg = np.asarray(img)
            if self.transform:
                transImg = self.transform(transImg)
            images.append(transImg)
            labels.append(int(self.videos_name[idx]))

        sample = {'images': images, 'labels': np.asarray(labels)}
        sample = Normalize_T(sample)
        return sample


if __name__ == '__main__':

    # 光流部分
    dataset = GolfDB_DIS_OPT()

    data_loader = DataLoader(dataset, batch_size=1,
                             shuffle=False, num_workers=6, drop_last=False)
    for i, sample in enumerate(data_loader):
        images, labels = sample['images'], sample['labels']
        print("video_file_name:{}  video_frames_len:{}".format(
            labels[0, 0], labels.shape[1]))

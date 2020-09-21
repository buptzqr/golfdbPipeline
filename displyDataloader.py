import os.path as osp
import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import sys

# data_info 文件格式为 文件所在位置的绝对路径
# 非光流部分


def ToTensor(sample):
    """Convert ndarrays in sample to Tensors."""
    images, labels = sample['images'], sample['labels']
    # images= np.asarray(images)
    nImages = []
    for image in images:
        if isinstance(image, np.ndarray):
            pass
        else:
            image = image.numpy()
        nImages.append(image)
        # nImages.append(image)
    images = np.asarray(nImages)
    images = images.transpose((0, 3, 1, 2))
    return {'images': torch.from_numpy(images).float().div(255.),
            'labels': torch.from_numpy(labels).long()}


def Normalize(sample, mean, std):
    images, labels = sample['images'], sample['labels']
    images.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
    return {'images': images, 'labels': labels}


class GolfDB(Dataset):
    def __init__(self, data_file, vid_dir, transform=None, myMean=[], myStd=[]):
        self.dataInfo = open(data_file)
        self.transform = transform
        self.myMean = myMean
        self.myStd = myStd
        self.videosPath = []  # 用来存储video对应的图片帧所在位置
        for line in self.dataInfo:
            line = line.rstrip()
            info = line.split('/')[-1]
            self.videosPath.append(os.path.join(vid_dir, info))

    def __len__(self):
        return len(self.videosPath)

    def __getitem__(self, idx):
        images = []
        labels = []
        videoPath = self.videosPath[idx]
        for imgName in os.listdir(videoPath):
            # full clip
            img = cv2.imread(os.path.join(videoPath, imgName))
            transImg = np.asarray(img)
            if self.transform:
                transImg = self.transform(transImg)
            images.append(transImg)
            labels.append(8)

        sample = {'images': images, 'labels': np.asarray(labels)}
        sample = ToTensor(sample)
        return sample


# 光流部分
def Normalize_T(sample):
    images, labels = sample['images'], sample['labels']
    images = np.asarray(images)
    labels = np.asarray(labels)
    # print(images.shape)
    imgsMean = np.mean(images, axis=(1, 2))
    imgsMean = imgsMean.reshape(-1, 1, 1, 3)
    images = np.subtract(images, imgsMean)
    images = images.transpose((0, 3, 1, 2))
    images = torch.from_numpy(images)
    labels = torch.from_numpy(labels)
    return {'images': images, 'labels': labels}


class GolfDB_T(Dataset):
    def __init__(self, data_file, transform=None):
        # self.df = pd.read_pickle(data_file)
        self.dataInfo = open(data_file)
        self.transform = transform
        self.opticalFilessPath = []  # 用来存储video对应的光流文件所在位置
        for line in self.dataInfo:
            line = line.rstrip()
            info = line.split()
            self.opticalFilessPath.append(info[0])

    def __len__(self):
        return len(self.opticalFilessPath)

    def __getitem__(self, idx):

        images, labels = [], []
        opticalFileFolder = self.opticalFilessPath[idx]
        # full clip
        # get files num
        filesNum = -1
        for ger in os.walk(opticalFileFolder):
            filesNum = len(ger[2])
        for pos in range(1, filesNum + 1):
            opticalFileName = osp.join(
                opticalFileFolder, '{:0>4d}.flo'.format(pos))
            # print(opticalFileName)
            opticalOri = np.fromfile(
                opticalFileName, np.float32, offset=12).reshape(160, 160, 2)
            opticalArray = np.empty([160, 160, 3], np.float32)
            opticalArray[..., 0] = 255
            opticalArray[..., 1] = opticalOri[:, :, 0]
            opticalArray[..., 2] = opticalOri[:, :, 1]
            # print(opticalFileName + "is ok")
            if self.transform:
                opticalArray = self.transform(opticalArray)
            images.append(opticalArray)
            labels.append(8)
        sample = {'images': images, 'labels': np.asarray(labels)}
        sample = Normalize_T(sample)

        return sample


if __name__ == '__main__':
    # 非光流部分
    # myMean=[0.485, 0.456, 0.406]
    # myStd=[0.229, 0.224, 0.225]  # ImageNet mean and std (RGB)

    # dataset = GolfDB(data_file,
    #                  transform=transforms.Compose([transforms.ToPILImage(),
    #                     transforms.RandomHorizontalFlip(0.5),
    #                     transforms.RandomAffine(5,shear=5),
    #                     transforms.ToTensor()]),
    #                  myMean=myMean,
    #                  myStd=myStd)

    data_file = "/home/zqr/codes/data/data_info.txt"
    # 光流部分
    dataset = GolfDB_T(data_file,
                       transform=None)

    data_loader = DataLoader(dataset, batch_size=1,
                             shuffle=False, num_workers=6, drop_last=False)

    for i, sample in enumerate(data_loader):
        images, labels = sample['images'], sample['labels']
        events = np.where(labels.squeeze() < 8)[0]
        print('{} events: {}'.format(len(events), events))

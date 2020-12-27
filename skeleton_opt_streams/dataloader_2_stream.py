import os.path as osp
import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import json


def Normalize_T(sample):
    images, keypoints, labels = sample['images'], sample['keypoints'], sample['labels']
    images = np.asarray(images)
    images = images.astype(np.float32)
    labels = np.asarray(labels)
    # print(images.shape)
    imgsMean = np.mean(images, axis=(1, 2))
    imgsMean = imgsMean.reshape(-1, 1, 1, 3)
    images = np.subtract(images, imgsMean)
    images = images.transpose((0, 3, 1, 2))
    images = torch.from_numpy(images)
    labels = torch.from_numpy(labels)
    keypoints = torch.from_numpy(keypoints)
    keypoints = keypoints.permute(1,0,2,3).contiguous()
    return {'images': images, 'keypoints': keypoints, 'labels': labels}


class GolfDB_2_Stream(Dataset):
    def __init__(self, data_file, vid_dir, seq_length, kp_path="/home/zqr/data/golfdb_keypoints/human_keypoints/simplify", node_num=16, transform=None, train=True):
        self.df = pd.read_pickle(data_file)
        self.vid_dir = vid_dir
        self.seq_length = seq_length
        self.transform = transform
        self.train = train
        self.kp_path = kp_path
        self.node_num = node_num

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        a = self.df.loc[idx, :]  # annotation info
        events = a["events"]
        # now frame #s correspond to frames in preprocessed video clips
        events -= events[0]

        images, labels, keypoints = [], [], []
        opticalFileFolder = osp.join(self.vid_dir, '{}'.format(a["id"]))
        kp_file = osp.join(self.kp_path, str(a["id"]) + ".json")
        kp_list = []
        with open(kp_file, 'r') as f:
            kp_data_list = json.load(f)
        kp_data_list.sort(key=lambda res: res["image_id"])
        x_data = []
        y_data = []
        z_data = []
        for elem in kp_data_list:
            kp_list.append(elem["keypoints"])
        video_len = len(kp_list)
        for i in range(video_len):
            for j in range(self.node_num):
                x_data.append(kp_list[i][j * 3])
                y_data.append(kp_list[i][j * 3 + 1])
                z_data.append(kp_list[i][j * 3 + 2])
        data_adj = [x_data, y_data, z_data]
        data_numpy = np.array(data_adj)
        data_numpy = data_numpy.reshape(3, video_len, self.node_num, 1)

        # print(opticalFileFolder)
        if self.train:
            start_frame = np.random.randint(events[-1] + 1)
            pos = start_frame
            # 光流文件是从第一帧开始的
            if pos == 0:
                pos = 1
            while len(images) < self.seq_length:
                opticalFileName = osp.join(
                    opticalFileFolder, '{:0>4d}.jpg'.format(pos))
                if os.path.exists(opticalFileName):
                    opticalArray = cv2.imread(opticalFileName)
                    # opticalOri = np.fromfile(
                    #     opticalFileName, np.float32, offset=12).reshape(160, 160, 2)
                    # opticalArray = np.empty([160, 160, 3], np.float32)
                    # opticalArray[..., 0] = 255
                    # opticalArray[..., 1] = opticalOri[:, :, 0]
                    # opticalArray[..., 2] = opticalOri[:, :, 1]

                    if self.transform:
                        opticalArray = self.transform(opticalArray)
                    images.append(opticalArray)
                    pos_adj = pos - 1
                    keypoints.append(data_numpy[:, pos_adj, :, :])

                    if pos_adj in events[1:-1]:
                        labels.append(np.where(events[1:-1] == pos_adj)[0][0])
                    else:
                        labels.append(8)
                    pos += 1
                else:
                    pos = 1
            keypoints = np.array(keypoints).swapaxes(0, 1)
        else:
            # full clip
            # get files num
            keypoints = data_numpy.astype(np.float32)
            filesNum = -1
            for ger in os.walk(opticalFileFolder):
                filesNum = len(ger[2])
            for pos in range(1, filesNum + 1):
                opticalFileName = osp.join(
                    opticalFileFolder, '{:0>4d}.jpg'.format(pos))
                opticalArray = cv2.imread(opticalFileName)
                # print(opticalFileName)
                # opticalOri = np.fromfile(
                #     opticalFileName, np.float32, offset=12).reshape(160, 160, 2)
                # opticalArray = np.empty([160, 160, 3], np.float32)
                # opticalArray[..., 0] = 255
                # opticalArray[..., 1] = opticalOri[:, :, 0]
                # opticalArray[..., 2] = opticalOri[:, :, 1]
                # print(opticalFileName + "is ok")
                if self.transform:
                    opticalArray = self.transform(opticalArray)
                images.append(opticalArray)
                pos_adj = pos - 1
                if pos_adj in events[1:-1]:
                    labels.append(np.where(events[1:-1] == pos_adj)[0][0])
                else:
                    labels.append(8)

        sample = {'images': images, 'keypoints': keypoints.astype(
            np.float32), 'labels': np.asarray(labels)}
        sample = Normalize_T(sample)

        return sample


if __name__ == '__main__':

    dataset = GolfDB_2_Stream(data_file='data/train_split_1.pkl',
                              vid_dir='/home/zqr/data/optical/opticalFlowOri_160',
                              seq_length=64,
                              train=True)

    data_loader = DataLoader(dataset, batch_size=1,
                             shuffle=False, num_workers=6, drop_last=False)
    count = 0
    for i, sample in enumerate(data_loader):
        images, keypoints, labels = sample['images'], sample['keypoints'], sample['labels']
        # print(labels)
        events = np.where(labels.squeeze() < 8)[0]
        count += 1
        print('{}-{} events: {}'.format(count, len(events), events))

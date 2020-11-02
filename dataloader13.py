import os.path as osp
import os
import cv2
import pandas as pd
import numpy as np
import torch
import json
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from data.config import cfg


def ToTensor_13(sample):
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


def Normalize_13(sample):
    myMean = [0.485, 0.456, 0.406]
    myStd = [0.229, 0.224, 0.225]  # ImageNet mean and std (RGB)

    mean = torch.tensor(myMean)
    std = torch.tensor(myStd)
    images, labels = sample['images'], sample['labels']
    images.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
    return {'images': images, 'labels': labels}


def Normalize_T_13(sample, flag):
    images, labels = sample['images'], sample['labels']
    nImages = []
    # if flag:
    #     for image in images:
    #         if isinstance(image, np.ndarray):
    #             pass
    #         else:
    #             image = image.numpy()
    #         nImages.append(image)
    #     images = np.asarray(nImages)
    #     images = images.transpose((0, 2, 3, 1))
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


class GolfDB_13(Dataset):
    def __init__(self, data_file, json_dir, dataloader_opt, seq_length, train=True):
        self.df = pd.read_pickle(data_file)
        self.json_dir = json_dir
        self.seq_length = seq_length
        self.dataloader_opt = dataloader_opt
        self.transform = None
        if self.dataloader_opt == cfg.DATALOADER_OPT.RGB:
            self.transform = transforms.Compose([transforms.ToPILImage(),
                                                 transforms.RandomHorizontalFlip(
                                                     0.5),
                                                 transforms.RandomAffine(
                                                     5, shear=5),
                                                 transforms.ToTensor()])
        # if self.dataloader_opt == cfg.DATALOADER_OPT.OPTICAL_FLOW:
        #     self.transform = transforms.Compose([transforms.ToPILImage(),
        #                                          transforms.RandomHorizontalFlip(
        #                                              0.5),
        #                                          transforms.ToTensor()])

        self.train = train
        self.json_files = []
        for file in os.listdir(self.json_dir):
            self.json_files.append(file)
        self.data_num = len(self.json_files)

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        file = self.json_files[idx]
        video_name = file[:11]
        with open(os.path.join(self.json_dir, file), 'r') as f:
            json_str = f.read()
            json_data = json.loads(json_str)
            json_id = int(file[12])
            videos_info = self.df[self.df["youtube_id"] == video_name]
            # 得到了id，也就是范围[0,1400)的数字,根据这个id去做好光流图和切好原始图的文件夹中读取数据
            video_id = str(videos_info.iloc[json_id]["id"])
            # 得到了对应的帧号
            events_list = np.array(json_data["categories"][0]["events"])
            # golfdb原始帧号标注
            t = videos_info.iloc[json_id]["events"]
            # 修正无法取到文件尾的问题
            final_frame = t[-1] - t[0]
            events_list = np.append(events_list, final_frame)
            # 获得数据文件的位置
            data_dir = ""
            if self.dataloader_opt == cfg.DATALOADER_OPT.OPTICAL_FLOW:
                data_dir = ""
                data_dir = data_dir + cfg.OPT_RESIZE_FILE_PATH
                images, labels = [], []
                opticalFileFolder = osp.join(
                    data_dir, '{}'.format(video_id))

                if self.train:
                    start_frame = np.random.randint(events_list[-1] + 1)
                    pos = start_frame

                    if pos == 0:
                        pos = 1
                    while len(images) < self.seq_length:
                        opticalFileName = osp.join(
                            opticalFileFolder, '{:0>4d}.jpg'.format(pos))
                        if os.path.exists(opticalFileName):
                            opticalArray = cv2.imread(opticalFileName)
                            if self.transform:
                                opticalArray = self.transform(opticalArray)
                            images.append(opticalArray)
                            # 光流文件编号是从1开始的，文件名为1对应的是第0帧的运动情况
                            # 这是我编号的问题，但是数据已经处理好了，所以代码中要纠正一下这个偏差
                            pos_adj = pos - 1
                            if pos_adj in events_list[0:-1]:
                                labels.append(
                                    np.where(events_list[0:-1] == pos_adj)[0][0])
                            else:
                                labels.append(13)
                            pos += 1
                        else:
                            pos = 1
                else:
                    # 视频中的所有帧
                    filesNum = -1
                    for ger in os.walk(opticalFileFolder):
                        filesNum = len(ger[2])
                    for pos in range(1, filesNum + 1):
                        opticalFileName = osp.join(
                            opticalFileFolder, '{:0>4d}.jpg'.format(pos))
                        opticalArray = cv2.imread(opticalFileName)
                        images.append(opticalArray)
                        pos_adj = pos - 1
                        if pos_adj in events_list[0:-1]:
                            labels.append(
                                np.where(events_list[0:-1] == pos_adj)[0][0])
                        else:
                            labels.append(13)
                sample = {'images': images, 'labels': np.asarray(labels)}
                sample = Normalize_T_13(sample, self.train)
                return sample

            if self.dataloader_opt == cfg.DATALOADER_OPT.RGB:
                data_dir = ""
                data_dir = data_dir + cfg.VIDEO_160_PATH

                images, labels = [], []
                cap = cv2.VideoCapture(
                    osp.join(data_dir, '{}.mp4'.format(video_id)))

                if self.train:
                    start_frame = np.random.randint(events_list[-1] + 1)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                    pos = start_frame
                    while len(images) < self.seq_length:
                        ret, img = cap.read()
                        if ret:
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            transImg = np.asarray(img)
                            if self.transform:
                                transImg = self.transform(transImg)
                            transImg = transImg.permute((1, 2, 0))
                            images.append(transImg)
                            if pos in events_list[0:-1]:
                                labels.append(
                                    np.where(events_list[0:-1] == pos)[0][0])
                            else:
                                labels.append(13)
                            pos += 1
                        else:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            pos = 0
                    cap.release()
                else:
                    # full clip
                    for pos in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
                        _, img = cap.read()
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        transImg = np.asarray(img)
                        images.append(transImg)
                        if pos in events_list[0:-1]:
                            labels.append(
                                np.where(events_list[0:-1] == pos)[0][0])
                        else:
                            labels.append(13)
                    cap.release()

                sample = {'images': images, 'labels': np.asarray(labels)}
                sample = ToTensor_13(sample)
                sample = Normalize_13(sample)
                return sample

            if self.dataloader_opt == cfg.DATALOADER_OPT.KEYPOINTS:
                # TODO:从points文件中读取keypoints然后
                print("we aren't ready for this")
                return


if __name__ == '__main__':

    dataset = GolfDB_13(data_file=cfg.OUR_PKL_FILE_PATH,
                        json_dir=cfg.VAL_JSON_PATH,
                        dataloader_opt=cfg.DATAOPT,
                        seq_length=64,
                        train=True)

    data_loader = DataLoader(dataset, batch_size=1,
                             shuffle=False, num_workers=6, drop_last=False)

    for i, sample in enumerate(data_loader):
        images, labels = sample['images'], sample['labels']
        events = np.where(labels.squeeze() < 13)[0]
        print('{} events: {}'.format(len(events), events))

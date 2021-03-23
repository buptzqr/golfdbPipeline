import os.path as osp
import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

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
    mean = torch.tensor(mean)
    std = torch.tensor(std)
    images, labels = sample['images'], sample['labels']
    images.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
    return {'images': images, 'labels': labels}

def Normalize_T(sample):
    images, labels = sample['opt_images'], sample['labels']
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
    return {'opt_images': images, 'labels': labels}


class GolfDB_2_stream(Dataset):
    def __init__(self, data_file, vid_dir, seq_length, transform=None,
                img_dir="/home/zqr/data/videos_160", img_transform = transforms.Compose([transforms.ToPILImage(),
                                                   transforms.RandomHorizontalFlip(
                                                       0.5),
                                                   transforms.RandomAffine(
                                                       5, shear=5),
                                                   transforms.ToTensor()]),
                                                    myMean = [0.485, 0.456, 0.406],
                                                    myStd = [0.229, 0.224, 0.225],
                                                    train=True):
        self.df = pd.read_pickle(data_file)
        self.vid_dir = vid_dir
        self.seq_length = seq_length
        self.transform = transform
        self.train = train
        self.img_transform = img_transform
        self.img_dir = img_dir
        self.myMean = myMean
        self.myStd = myStd
        

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        a = self.df.loc[idx, :]  # annotation info
        events = a['events']
        # now frame #s correspond to frames in preprocessed video clips
        events -= events[0]

        images,opt_images, labels = [], [], []
        opticalFileFolder = osp.join(self.vid_dir, '{}'.format(a['id']))
        cap = cv2.VideoCapture(
            osp.join(self.img_dir, '{}.mp4'.format(a['id'])))
        # print(opticalFileFolder)
        if self.train:
            start_frame = np.random.randint(events[-1] + 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            pos = start_frame
            # 光流文件是从第一帧开始的
            if pos == 0:
                pos = 1
            while len(opt_images) < self.seq_length:
                ret, img = cap.read()
                opticalFileName = osp.join(
                    opticalFileFolder, '{:0>4d}.jpg'.format(pos))
                if os.path.exists(opticalFileName) and ret:
                    #读取rgb文件
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    transImg = np.asarray(img)
                    if self.img_transform:
                        transImg = self.img_transform(transImg)
                    transImg = transImg.permute((1, 2, 0))
                    images.append(transImg)
                    #读取光流文件
                    opticalArray = cv2.imread(opticalFileName)
                    if self.transform:
                        opticalArray = self.transform(opticalArray)
                    opt_images.append(opticalArray)
                    pos_adj = pos - 1

                    if pos_adj in events[1:-1]:
                        labels.append(np.where(events[1:-1] == pos_adj)[0][0])
                    else:
                        labels.append(8)
                    pos += 1
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    pos = 1
            cap.release()
        else:
            # 添加rgb
            for pos in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
                _, img = cap.read()
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                transImg = np.asarray(img)
                images.append(transImg)
            cap.release()
            # full clip
            # get files num
            # 添加光流图和label
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
                opt_images.append(opticalArray)
                pos_adj = pos - 1
                if pos_adj in events[1:-1]:
                    labels.append(np.where(events[1:-1] == pos_adj)[0][0])
                else:
                    labels.append(8)
        sample_opt = {'opt_images': opt_images, 'labels': np.asarray(labels)}
        sample_opt = Normalize_T(sample_opt)
        
        
        sample_rgb = {'images': images, 'labels': np.asarray(labels)}
        sample_rgb = ToTensor(sample_rgb)
        sample_rgb = Normalize(sample_rgb, self.myMean, self.myStd)
        
        sample = {'images':sample_rgb['images'],'opt_images':sample_opt['opt_images'],'labels':sample_opt['labels']}
        return sample


if __name__ == '__main__':

    dataset = GolfDB_2_stream(data_file='data/train_split_1.pkl',
                       vid_dir='/home/zqr/data/optical/opticalFlowOri_160',
                       seq_length=64,
                       train=True)

    data_loader = DataLoader(dataset, batch_size=1,
                             shuffle=False, num_workers=6, drop_last=False)
    count = 0
    for i, sample in enumerate(data_loader):
        images, opt_images,labels = sample['images'], sample['opt_images'], sample['labels']
        # print(labels)
        events = np.where(labels.squeeze() < 8)[0]
        count += 1
        print('{}-{} events: {}'.format(count, len(events), events))

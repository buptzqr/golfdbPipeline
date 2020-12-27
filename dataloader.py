import os.path as osp
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


class GolfDB(Dataset):
    def __init__(self, data_file, vid_dir, seq_length, transform=None, train=True, myMean=[], myStd=[]):
        self.df = pd.read_pickle(data_file)
        self.vid_dir = vid_dir
        self.seq_length = seq_length
        self.transform = transform
        self.train = train
        self.myMean = myMean
        self.myStd = myStd

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        a = self.df.loc[idx, :]  # annotation info
        # 这样做是因为处理后的1400视频就是截取的events范围内的视频
        events = a['events']
        # now frame #s correspond to frames in preprocessed video clips
        events -= events[0]

        images, labels = [], []
        cap = cv2.VideoCapture(
            osp.join(self.vid_dir, '{}.mp4'.format(a['id'])))

        if self.train:
            # random starting position, sample 'seq_length' frames
            start_frame = np.random.randint(events[-1] + 1)
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
                    if pos in events[1:-1]:
                        labels.append(np.where(events[1:-1] == pos)[0][0])
                    else:
                        labels.append(8)
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
                # eval不需要做各种仿射变换
                # if self.transform:
                #     transImg=self.transform(transImg)
                images.append(transImg)
                if pos in events[1:-1]:
                    labels.append(np.where(events[1:-1] == pos)[0][0])
                else:
                    labels.append(8)
            cap.release()

        sample = {'images': images, 'labels': np.asarray(labels)}
        # sample = {'images':np.asarray(images), 'labels':np.asarray(labels)}
        # if self.transform:
        #     sample['images'] = self.transform(sample['images'])
        sample = ToTensor(sample)
        sample = Normalize(sample, self.myMean, self.myStd)
        # print(osp.join(self.vid_dir, '{}.mp4'.format(a['id'])))
        # print(sample['images'].shape)
        # print(sample['labels'].shape)
        return sample


if __name__ == '__main__':

    myMean = [0.485, 0.456, 0.406]
    myStd = [0.229, 0.224, 0.225]  # ImageNet mean and std (RGB)

    dataset = GolfDB(data_file='data/train_split_1.pkl',
                     vid_dir='/home/zqr/data/videos_160',
                     seq_length=64,
                     transform=transforms.Compose([transforms.ToPILImage(),
                                                   transforms.RandomHorizontalFlip(
                                                       0.5),
                                                   transforms.RandomAffine(
                                                       5, shear=5),
                                                   transforms.ToTensor()]),
                     train=True,
                     myMean=myMean,
                     myStd=myStd)

    data_loader = DataLoader(dataset, batch_size=1,
                             shuffle=False, num_workers=6, drop_last=False)

    for i, sample in enumerate(data_loader):
        images, labels = sample['images'], sample['labels']
        events = np.where(labels.squeeze() < 8)[0]
        print('{} events: {}'.format(len(events), events))

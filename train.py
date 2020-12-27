from dataloader import GolfDB, Normalize, ToTensor
from dataloader_T import GolfDB_T, Normalize_T
from model import EventDetector
from util import *
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from torch.utils.tensorboard import SummaryWriter
from myeval import myeval
from data.config import cfg
import torch.nn as nn


if __name__ == '__main__':

    # training configuration
    split = cfg.SPLIT
    iterations = cfg.ITERATIONS
    it_save = cfg.IT_SAVE  # save model every 100 iterations
    n_cpu = cfg.CPU_NUM
    seq_length = cfg.SEQUENCE_LENGTH
    bs = cfg.BATCH_SIZE  # batch size
    k = 10  # frozen layers

    model = EventDetector(pretrain=True,
                          width_mult=1.,
                          lstm_layers=1,
                          lstm_hidden=256,
                          bidirectional=True,
                          dropout=False)
    freeze_layers(k, model)
    model.train()
    model.cuda()
    model = nn.DataParallel(model)
    
    # 用来训练非光流部分
    # dataset = GolfDB(data_file='data/train_split_{}.pkl'.format(split),
    #                  vid_dir='data/videos_160/',
    #                  seq_length=seq_length,
    #                  transform=transforms.Compose([transforms.ToPILImage(),
    #                     transforms.RandomHorizontalFlip(0.5),
    #                     transforms.RandomAffine(5,shear=5),
    #                     transforms.ToTensor()]),
    #                  myMean=[0.485, 0.456, 0.406],
    #                  myStd=[0.229, 0.224, 0.225],
    #                  train=True)

    # 用来训练光流部分
    dataset = GolfDB_T(data_file='data/train_split_{}.pkl'.format(split),
                       vid_dir=cfg.OPT_RESIZE_FILE_PATH,
                       seq_length=seq_length,
                       train=True)

    data_loader = DataLoader(dataset,
                             batch_size=bs,
                             shuffle=True,
                             num_workers=n_cpu,
                             drop_last=True)

    # the 8 golf swing events are classes 0 through 7, no-event is class 8
    # the ratio of events to no-events is approximately 1:35 so weight classes accordingly:
    weights = torch.FloatTensor(
        [1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/35]).cuda()
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

    losses = AverageMeter()

    # writer = SummaryWriter()

    if not os.path.exists('models'):
        os.mkdir('models')

    i = 0
    while i < iterations:
        # for p in optimizer.param_groups:
        #     print(p['lr'])
        for sample in data_loader:
            images, labels = sample['images'].cuda(), sample['labels'].cuda()
            logits = model(images)
            labels = labels.view(bs*seq_length)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            losses.update(loss.item(), images.size(0))
            optimizer.step()
            print('Iteration: {}\tLoss: {loss.val:.4f} ({loss.avg:.4f})'.format(
                i, loss=losses))
            i += 1
            if i % it_save == 0:
                torch.save({'optimizer_state_dict': optimizer.state_dict(),
                            'model_state_dict': model.state_dict()}, 'models/swingnet_{}.pth.tar'.format(i))
            if i == 1000:
                for p in optimizer.param_groups:
                    p['lr'] *= 0.1
            if i == 2000:
                for p in optimizer.param_groups:
                    p['lr'] *= 0.1
            if i == iterations:
                break

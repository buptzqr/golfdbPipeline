from dataloader13 import GolfDB_13, ToTensor_13, Normalize_13, Normalize_T_13
from model import EventDetector
from util import *
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from torch.utils.tensorboard import SummaryWriter
from myeval import myeval
from data.config import cfg


if __name__ == '__main__':

    # training configuration
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

    dataset = GolfDB_13(data_file=cfg.OUR_PKL_FILE_PATH,
                        json_dir=cfg.TRAIN_JSON_PATH,
                        dataloader_opt=cfg.DATAOPT,
                        seq_length=64,
                        train=True)

    data_loader = DataLoader(dataset, batch_size=bs,
                             shuffle=True, num_workers=n_cpu, drop_last=True)

    weights = torch.FloatTensor(
        [1 / 13, 1 / 13, 1 / 13, 1 / 13, 1 / 13, 1 / 13, 1 / 13, 1 / 13, 1 / 13, 1 / 13, 1 / 13, 1 / 13, 1 / 13, 1 / 35]).cuda()

    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

    losses = AverageMeter()

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
            if i == 3000:
                for p in optimizer.param_groups:
                    p['lr'] *= 0.1
            if i == 6000:
                for p in optimizer.param_groups:
                    p['lr'] *= 0.1
            if i == iterations:
                break

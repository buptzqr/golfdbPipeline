from model import EventDetector
import torch
from torch.utils.data import DataLoader
from displyDataloader import GolfDB_DIS_OPT, Normalize_T
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
from util import correct_preds
import matplotlib.pyplot as plt
from data.config import cfg
import os
import json
import sys

def eval(model, seq_length, n_cpu, disp):

    # # 光流部分
    dataset = GolfDB_DIS_OPT()

    data_loader = DataLoader(dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=n_cpu,
                             drop_last=False)

    preds_collect = {}
    scores_collect = {}

    for i, sample in enumerate(data_loader):
        preds = []
        scores = []
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
        if eight_flag:
            for idx in range(8):
                pred = np.argsort(probs[:, idx])[-1]
                preds.append(pred.tolist())
                scores.append(probs[pred,idx].tolist())
        else:
            for idx in range(13):
                pred = np.argsort(probs[:, idx])[-1]
                preds.append(pred.tolist())
                scores.append(probs[pred,idx].tolist())
        preds_collect[labels[0, 0].item()] = preds
        scores_collect[labels[0, 0].item()] = scores
        
    return preds_collect,scores_collect


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("choose 8 or 13")
        sys.exit(1)
    eight_flag = False
    if sys.argv[1] == 'True':
        eight_flag = True
    
    seq_length = cfg.SEQUENCE_LENGTH
    n_cpu = cfg.CPU_NUM

    if eight_flag:
        save_dict = torch.load(cfg.TEST_MODEL_8)
        cfg.set_8_flag(True)
    else:
        cfg.set_8_flag(False)
        save_dict = torch.load(cfg.TEST_MODEL_13)
    model = EventDetector(pretrain=True,
                          width_mult=1.,
                          lstm_layers=1,
                          lstm_hidden=256,
                          bidirectional=True,
                          dropout=False)

    model.load_state_dict(save_dict['model_state_dict'])
    model.cuda()
    model.eval()
    preds,scores = eval(model, seq_length, n_cpu, True)
    
    # print(preds)
    for k, v in preds.items():
        src_path = os.path.join(cfg.TEST_IMGS_DIR, str(k))
        if eight_flag:
            dst_path = os.path.join(cfg.TEST_RESULT_TMP,"tmp_result_8",str(k))
        else:
            dst_path = os.path.join(cfg.TEST_RESULT_TMP,"tmp_result_13",str(k))
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)
        if eight_flag:
            for i in range(8):
                src_img = os.path.join(src_path, "{:0>4d}.jpg".format(v[i]))
                dst_img = os.path.join(
                    dst_path, "{:0>4d}_{}.jpg".format(i, v[i]))
                os.system("cp {} {}".format(src_img, dst_img))
        else:
            for i in range(13):
                src_img = os.path.join(src_path, "{:0>4d}.jpg".format(v[i]))
                dst_img = os.path.join(
                    dst_path, "{:0>4d}_{}.jpg".format(i, v[i]))
                os.system("cp {} {}".format(src_img, dst_img))
    
    # 把score写入文件中
    if not os.path.exists(cfg.TEST_SCORES_DIR):
            os.makedirs(cfg.TEST_SCORES_DIR)
    if eight_flag:
        with open(os.path.join(cfg.TEST_SCORES_DIR,"scores_8.json"),'w') as f:
            json.dump(scores,f) 
        with open(os.path.join(cfg.TEST_SCORES_DIR,"preds_8.json"),'w') as f:
            json.dump(preds,f) 
    else:           
        with open(os.path.join(cfg.TEST_SCORES_DIR,"scores_13.json"),'w') as f:
            json.dump(scores,f)
        with open(os.path.join(cfg.TEST_SCORES_DIR,"preds_13.json"),'w') as f:
            json.dump(preds,f) 
    
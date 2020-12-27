# 用来对李总的数据进行评价（你先需要对这个数据跑一遍pipeline，然后将得到的结果和对应的json中的ground_truth进行比较，从而进行评价）
import os
from data.config import cfg
import json
import numpy as np
if __name__ == "__main__":
    correct_list = []
    dir_list = []
    PCE = 0
    cfg.POST_EVAL_8 = True
    for res_dir in os.listdir(cfg.TEST_RESULT_PATH):
        json_path = os.path.join(
            cfg.TEST_JSON_PATH, "{}.json".format(res_dir))
        with open(json_path, 'r') as f:
            json_str = json.load(f)
        ground_truth = json_str["categories"][0]["events"]
        if cfg.POST_EVAL_8:
            ground_8 = [ground_truth[0], ground_truth[2], ground_truth[3], ground_truth[4],
                        ground_truth[5], ground_truth[8], ground_truth[10], ground_truth[12]]
            ground_truth = ground_8
        ground_truth = np.array(ground_truth)
        preds = []
        res_dir_abs_path = os.path.join(cfg.TEST_RESULT_PATH, res_dir)
        imgs = []
        for img in os.listdir(res_dir_abs_path):
            imgs.append(img)
        imgs.sort()
        for img in imgs:
            pred = int(img.split(".")[0].split('_')[1])
            preds.append(pred)
        preds = np.array(preds)
        if cfg.POST_EVAL_8:
            tol = int(max(np.round((preds[5] - preds[0]) / 30), 1))
        else:
            tol = int(max(np.round((preds[8] - preds[0]) / 30), 1))
        deltas = np.abs(ground_truth - preds)
        correct = (deltas <= tol).astype(np.uint8)
        correct_list.append(correct)
        dir_list.append(res_dir)
        print("{}:{}".format(res_dir, ground_truth))
    PCE = np.mean(correct_list)
    print(PCE)

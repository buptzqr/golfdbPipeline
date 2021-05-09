# 得到每张图片包括球杆在内的bbox和球杆关键点
# 每个文件夹下的bbox.txt存储的是每张图片包括杆在内bbox
# total_bbox存储的是总体的bbox
from detectron2.utils.visualizer import ColorMode
import torch
import random
import cv2
import data.config
from detectron2.engine import DefaultTrainer
import os
import pandas as pd
from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from google.colab.patches import cv2_imshow
import numpy as np
import detectron2
from detectron2.utils.logger import setup_logger
import json
setup_logger()
# import some common libraries

# import some common detectron2 utilities
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(
    "COCO-Keypoints/keypoint_rcnn_R_50_FPN_1x.yaml"))
# path to the model we just trained
cfg.MODEL.WEIGHTS = data.config.cfg.BBOX_CLUB_POINTS_MODEL_PATH
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8   # set a custom testing threshold
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (club)
cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 2  # only have 2 point head and tail
predictor = DefaultPredictor(cfg)
GOLFDB_PKL_PATH = "/home/zqr/codes/MyGolfDB/data/golfDB.pkl"
df = pd.read_pickle(GOLFDB_PKL_PATH)


def if_intersect(*detect_bbox):
    d_x1 = detect_bbox[0]
    d_y1 = detect_bbox[1]
    d_x2 = detect_bbox[2]+detect_bbox[0]
    d_y2 = detect_bbox[3]+detect_bbox[1]
    g_x1 = detect_bbox[4]
    g_y1 = detect_bbox[5]
    g_x2 = detect_bbox[6]+detect_bbox[4]
    g_y2 = detect_bbox[7]+detect_bbox[5]
    f_x1 = max(d_x1, g_x1)
    f_y1 = max(d_y1, g_y1)
    f_x2 = min(d_x2, g_x2)
    f_y2 = min(d_y2, g_y2)
    if f_x1 < f_x2 and f_y1 < f_y2:
        return True
    else:
        return False


if __name__ == '__main__':
    img_dirs = "/home/zqr/data/golfdb_frame_no_resize"
    if data.config.cfg.TEST_FLAG:
        img_dirs = data.config.cfg.TEST_IMGS_DIR
    img_dirs = []
    for dir_name in os.listdir(data.config.cfg.TEST_VIDEO_PAHT):
        img_dirs.append(dir_name.split('.')[0])
    for img_dir in img_dirs:
        if not data.config.cfg.TEST_FLAG:
            golfdb_bbox = df.iloc[int(img_dir)]["bbox"]
        print("begin process dir {}".format(img_dir))
        bbox_info_path = os.path.join(
            "/home/zqr/data/golfdb_keypoints/club_keypoints/bbox", img_dir)
        club_keypoints_path = os.path.join(
            "/home/zqr/data/golfdb_keypoints/club_keypoints/result", img_dir)
        if data.config.cfg.TEST_FLAG:
            bbox_info_path = os.path.join(
                data.config.cfg.TEST_BBOX_INFO_PATH, img_dir)
            club_keypoints_path = os.path.join(
                data.config.cfg.TEST_CLUB_KEYPOINTS_PATH, img_dir)

        if not os.path.exists(bbox_info_path):
            os.mkdir(bbox_info_path)
        else:
            continue
        if not os.path.exists(club_keypoints_path):
            os.mkdir(club_keypoints_path)

        img_dir_abs_path = os.path.join(img_dirs, img_dir)
        total_bbox = []
        all_keypoints = []
        flag = True
        for img in os.listdir(img_dir_abs_path):
            img_abs_path = os.path.join(img_dir_abs_path, img)
            img_path = os.path.join(img_dir_abs_path, img)
            im = cv2.imread(img_path)
            if not data.config.cfg.TEST_FLAG:
                # 这样做是因为数据集存在两个连着的人挥杆的情况，测试情况下不需要
                if flag:
                    img_width = im.shape[1]
                    img_height = im.shape[0]
                    x1 = int(img_width * golfdb_bbox[0])
                    y1 = int(img_height * golfdb_bbox[1])
                    w = int(img_width * golfdb_bbox[2])
                    h = int(img_height * golfdb_bbox[3])
                    x2 = x1 + w
                    y2 = y1 + h
                im = im[y1:y2, x1:x2]
            outputs = predictor(im)
            instances = outputs["instances"]
            scores = instances.get("scores")
            if len(scores):
                keypoints_info = {}
                max_score_idx = scores.argmax().item()
                max_instance = instances.__getitem__(max_score_idx)
                bbox = max_instance.get(
                    "pred_boxes").tensor[0].tolist()  # bbox 格式是x1y1x2y2
                if not data.config.cfg.TEST_FLAG:
                    # 因为只截取了一部分图片作为输入，所以需要将他预测的坐标还原回全局
                    bbox[0] += x1
                    bbox[1] += y1
                    bbox[2] += x1
                    bbox[3] += y1
                if flag:
                    total_bbox = bbox
                    flag = False
                else:
                    if total_bbox[0] > bbox[0]:
                        total_bbox[0] = bbox[0]
                    if total_bbox[1] > bbox[1]:
                        total_bbox[1] = bbox[1]
                    if total_bbox[2] < bbox[2]:
                        total_bbox[2] = bbox[2]
                    if total_bbox[3] < bbox[3]:
                        total_bbox[3] = bbox[3]

                keypoints = torch.squeeze(
                    max_instance.get("pred_keypoints")).tolist()
                if not data.config.cfg.TEST_FLAG:
                    # 关键点也得还原
                    keypoints[0][0] += x1
                    keypoints[0][1] += y1
                    keypoints[1][0] += x1
                    keypoints[1][1] += y1
                keypoints_info["image_id"] = img_dir + "/" + img
                keypoints_info["keypoints"] = keypoints
                all_keypoints.append(keypoints_info)

                with open(os.path.join(bbox_info_path, "bbox.txt"), 'a') as f:
                    f.write(img_abs_path + ":" + str(bbox) + "\n")
        with open(os.path.join(bbox_info_path, "total_bbox.txt"), 'w') as f:
            f.write(img_dir_abs_path + ":" + str(total_bbox) + "\n")
        with open(os.path.join(club_keypoints_path, "club_keypoints.json"), "w") as f:
            json.dump(all_keypoints, f)
        print("finish process dir {}".format(img_dir))

    print("get info ok")

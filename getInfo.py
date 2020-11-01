# 得到每张图片包括球杆在内的bbox和球杆关键点
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

if __name__ == '__main__':
    img_dirs = data.config.cfg.TEST_IMGS_DIR
    for img_dir in os.listdir(img_dirs):
        bbox_info_path = os.path.join(
            data.config.cfg.TEST_BBOX_INFO_PATH, img_dir)
        club_keypoints_path = os.path.join(
            data.config.cfg.TEST_CLUB_KEYPOINTS_PATH, img_dir)
        if not os.path.exists(bbox_info_path):
            os.mkdir(bbox_info_path)
        if not os.path.exists(club_keypoints_path):
            os.mkdir(club_keypoints_path)
        img_dir_abs_path = os.path.join(img_dirs, img_dir)
        for img in os.listdir(img_dir_abs_path):
            img_abs_path = os.path.join(img_dir_abs_path, img)
            img_path = os.path.join(img_dir_abs_path, img)
            im = cv2.imread(img_path)
            outputs = predictor(im)
            instances = outputs["instances"]
            scores = instances.get("scores")
            if len(scores):
                max_score_idx = scores.argmax().item()
                max_instance = instances.__getitem__(max_score_idx)
                bbox = max_instance.get("pred_boxes").tensor[0].tolist()
                keypoints = torch.squeeze(
                    max_instance.get("pred_keypoints")).tolist()
                with open(os.path.join(club_keypoints_path, "club_keypoints.txt"), 'a') as f:
                    f.write(img_abs_path+":"+str(keypoints)+"\n")
                with open(os.path.join(bbox_info_path, "bbox.txt"), 'a') as f:
                    f.write(img_abs_path + ":" + str(bbox) + "\n")
    print("get info ok")

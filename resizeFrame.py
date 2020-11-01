# 用来根据bbox resize图片到指定大小
import pandas as pd
import os
import cv2
from multiprocessing import Pool
import numpy as np
import sys
from data.config import cfg
if __name__ == '__main__':
    img_top_dir = cfg.TEST_IMGS_DIR  # 为节省空间直接覆盖原视频帧
    bbox_top_dir = cfg.TEST_BBOX_INFO_PATH
    dim = cfg.RESIZE_DIM
    # img_res_dir = "/home/zqr/data/test/test_opt"
    for dir in os.listdir(img_top_dir):
        print("resize folder:{}".format(dir))
        img_dir_abs_path = os.path.join(img_top_dir, dir)
        bbox_abs_path = os.path.join(bbox_top_dir, dir, "bbox.txt")
        bbox_map = {}
        # imagesFoler = os.path.join(img_res_dir, dir)
        # if not os.path.exists(imagesFoler):
        #     os.makedirs(imagesFoler)
        with open(bbox_abs_path, 'r') as f:
            for line in f.readlines():
                bbox = []
                line = line.rstrip()
                bboxKey = line.split(':')[0].split('/')[-1]
                tmp = line.split(':', 1)[1]
                tmp = tmp.rstrip(']')
                tmp = tmp.strip('[')
                a1 = tmp.split(',', 3)[0]
                a2 = tmp.split(',', 3)[1]
                a3 = tmp.split(',', 3)[2]
                a4 = tmp.split(',', 3)[3]
                bbox.append(float(a1))
                bbox.append(float(a2))
                bbox.append(float(a3))
                bbox.append(float(a4))
                bbox_map[bboxKey] = bbox
        for img in os.listdir(img_dir_abs_path):
            if img not in bbox_map.keys():
                pass
            else:
                bbox = bbox_map[img]
                x = int(bbox[0])
                y = int(bbox[1])
                w = int(bbox[2] - bbox[0])
                h = int(bbox[3] - bbox[1])
                img_abs_path = os.path.join(img_dir_abs_path, img)
                image = cv2.imread(img_abs_path)
                crop_img = image[y:y + h, x:x + w]
                crop_size = crop_img.shape[:2]
                ratio = dim / max(crop_size)
                new_size = tuple([int(x*ratio) for x in crop_size])
                resized = cv2.resize(crop_img, (new_size[1], new_size[0]))
                delta_w = dim - new_size[1]
                delta_h = dim - new_size[0]
                top, bottom = delta_h // 2, delta_h - (delta_h // 2)
                left, right = delta_w // 2, delta_w - (delta_w // 2)
                b_img = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                           value=[0.406*255, 0.456*255, 0.485*255])  # ImageNet means (BGR)
                cv2.imwrite(os.path.join(img_dir_abs_path, img), b_img)
    print("resize ok")

import cv2
import numpy as np
import json
from data.config import cfg
import os

import shutil
def visualize(keypoint_num, img, joints, score=None):
        pairs = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                 [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                 [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7],[18,19]]
        # 鼻子1，左眼2，右眼3，左耳4，右耳5，左肩6，右肩7，左肘8，右肘9，左腕10，
        # 右腕11，左臀12，右臀13，左膝14，右膝15，左踝16，右踝17，杆头18，杆尾19

        color = np.random.randint(0, 256, (keypoint_num, 3)).tolist()
        joints_array = np.ones((keypoint_num, 2), dtype=np.float32)
        for i in range(keypoint_num):
            joints_array[i, 0] = joints[i * 3]
            joints_array[i, 1] = joints[i * 3 + 1]
            # joints_array[i, 2] = joints[i * 3 + 2]

        def draw_line(img, p1, p2):
            c = (0, 0, 255)
            if p1[0] > 0 and p1[1] > 0 and p2[0] > 0 and p2[1] > 0:
                cv2.line(img, tuple(p1), tuple(p2), c, 2)

        for pair in pairs:
            draw_line(img, joints_array[pair[0] - 1],
                      joints_array[pair[1] - 1])

        for i in range(keypoint_num):
            if joints_array[i, 0] > 0 and joints_array[i, 1] > 0:
                cv2.circle(img, tuple(
                    joints_array[i, :2]), 2, (0,255,0), 2)

        return img

if __name__ == '__main__':
    for dir_name in os.listdir(cfg.TEST_RESULT_PATH):
        img_only_keypoints = "/home/zqr/data/test/result_only_keypoints"
        img_only_keypoints = os.path.join(img_only_keypoints,dir_name)
        img_with_keypoints = os.path.join(cfg.TEST_RESULT_WITH_KEYPONTS,dir_name)
        
        if os.path.exists(img_only_keypoints):
            shutil.rmtree(img_only_keypoints)
            os.makedirs(img_only_keypoints)
        if os.path.exists(img_with_keypoints):
            shutil.rmtree(img_with_keypoints)
            os.makedirs(img_with_keypoints)
        for img_name in os.listdir(os.path.join(cfg.TEST_RESULT_PATH,dir_name)):
            ori_img_name = img_name
            img_name = int(img_name.split('_')[1].split('.')[0])
            img_name = "{:0>4d}".format(img_name)+".jpg"
            img_name = dir_name + '/' + img_name
            club_keypoints = []
            all_keypoints = []
            with open(os.path.join(cfg.TEST_CLUB_KEYPOINTS_PATH,dir_name,"club_keypoints.json"),'r') as f:
                club_json_str = f.read()
                club_json_data = json.loads(club_json_str)
            with open(os.path.join(cfg.TEST_KEYPOINTS_PATH,"keypoints_result",dir_name,"results.json"),'r') as f:
                human_json_str = f.read()
                human_json_data = json.loads(human_json_str)
                
            for item in club_json_data:
                if item['image_id']== img_name:
                    for i in item['keypoints']:
                        club_keypoints.extend(i)
            for item in human_json_data:
                if item['image_id']== img_name:
                    all_keypoints = item['keypoints']
            all_keypoints.extend(club_keypoints)
            data_numpy = cv2.imread(os.path.join(
                    cfg.TEST_RESULT_PATH, dir_name,ori_img_name), cv2.IMREAD_COLOR)
            img = visualize(19,data_numpy,all_keypoints)
            cv2.imwrite(os.path.join(img_with_keypoints, ori_img_name), img)
            # 老王还要只有关键点的图片
            img_backgroud = visualize(19,np.zeros(img.shape,np.uint8),all_keypoints)
            cv2.imwrite(os.path.join(img_only_keypoints, ori_img_name), img_backgroud)
             
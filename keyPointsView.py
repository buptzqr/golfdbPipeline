import numpy as np
import cv2
import os
import json
# 用来可视化关键点(包括球杆，目前选择杆尾和左右手腕相连)

total_keypoints_path = "/home/zqr/data/test/total_keypoints"
img_dir_path = "/home/zqr/data/test/test_img"
present_dir_path = "/home/zqr/data/test/present"


def visualize(img, joints, score=None):
    keypoint_num = 19
    pairs = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
             [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
             [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7], [10, 19], [11, 19], [18, 19]]
    # 鼻子，左眼，右眼，左耳，右耳，左肩，右肩，左肘，右肘，左腕，右腕，左臀，右臀，左膝，右膝，左踝，右踝，杆头，杆尾
    color = np.random.randint(0, 256, (keypoint_num, 3)).tolist()
    joints_array = np.ones((keypoint_num, 2), dtype=np.float32)
    for i in range(keypoint_num):
        joints_array[i, 0] = joints[i * 3]
        joints_array[i, 1] = joints[i * 3 + 1]

    for i in range(keypoint_num):
        if joints_array[i, 0] > 0 and joints_array[i, 1] > 0:
            cv2.circle(img, tuple(
                joints_array[i, :2]), 2, tuple(color[i]), 2)

    def draw_line(img, p1, p2):
        c = (0, 0, 255)
        if p1[0] > 0 and p1[1] > 0 and p2[0] > 0 and p2[1] > 0:
            cv2.line(img, tuple(p1), tuple(p2), c, 2)

    for pair in pairs:
        draw_line(img, joints_array[pair[0] - 1],
                  joints_array[pair[1] - 1])

    return img


if __name__ == "__main__":
    for keypoints_json in os.listdir(total_keypoints_path):
        result_dir = keypoints_json.split(".")[0]
        print("process folder:{}".format(result_dir))
        result_dir_abs_path = os.path.join(present_dir_path, result_dir)
        if not os.path.exists(result_dir_abs_path):
            os.mkdir(result_dir_abs_path)
        keypoints_json_abs_path = os.path.join(
            total_keypoints_path, keypoints_json)
        with open(keypoints_json_abs_path, "r") as f:
            keypoints_list = json.load(f)
        for e in keypoints_list:
            img_abs_path = os.path.join(img_dir_path, e["image_id"])
            data_numpy = cv2.imread(img_abs_path, cv2.IMREAD_COLOR)
            img = visualize(data_numpy, e["keypoints"], e["score"])
            cv2.imwrite(os.path.join(present_dir_path, e['image_id']), img)
        print("finish folder:{}".format(result_dir))
        print("********************************")

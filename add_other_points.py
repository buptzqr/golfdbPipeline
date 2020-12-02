import os
import json
from enum import Enum
# 给已经处理好关键点文件添加肩部的中点（一个新关键点）
# 这个脚本写的一般，这种覆盖式的处理虽然节省了空间，但是将原始点信息搞混乱了
total_keypoints_path = "/home/zqr/data/test/total_keypoints"
# total_keypoints_path = "/home/zqr/data/golfdb_keypoints/all_keypoints"
# 头，左右肩，左右肘，左右腕,左右髋，左右膝，左右踝
P = [0.0706, 0.1374, 0.1374, 0.029, 0.029, 0.018,
     0.018, 0.1587, 0.1587, 0.0816, 0.0816, 0.03, 0.03]


class GRAVITY_CAL_OPT(Enum):
    MEAN = 0
    COEFFICIENT = 1


GRAVITY_OPT = GRAVITY_CAL_OPT.COEFFICIENT

if __name__ == '__main__':
    for json_file in os.listdir(total_keypoints_path):
        print("process file:{}".format(json_file))
        json_file_abs_path = os.path.join(total_keypoints_path, json_file)
        with open(json_file_abs_path, 'r') as f:
            keypoints_list = json.load(f)
            point_num = 17
            head_num = 5
        for elem in keypoints_list:
            all_points_x = 0
            all_points_y = 0
            all_points_score = 0
            if GRAVITY_OPT == GRAVITY_CAL_OPT.MEAN:
                # 计算身体的重心，即所有点的平均值(除杆头，杆尾)
                for j in range(point_num):
                    all_points_x += elem["keypoints"][3 * j]
                    all_points_y += elem["keypoints"][3 * j + 1]
                    all_points_score += elem["keypoints"][3 * j + 2]

            # 采用布拉温-舍菲尔模型计算人体重心
            # 计算头部点（取左右眼，左右耳，和鼻子的点平均值）
            if GRAVITY_OPT == GRAVITY_CAL_OPT.COEFFICIENT:
                point_num = 13
                # 计算头部点
                head_point_x = 0
                head_point_y = 0
                head_point_score = 0
                for j in range(head_num):
                    head_point_x += elem["keypoints"][3 * j]
                    head_point_y += elem["keypoints"][3 * j + 1]
                    head_point_score += elem["keypoints"][3 * j + 2]
                head_point_x = head_point_x / head_num
                head_point_y = head_point_y / head_num
                head_point_score = head_point_score / head_num
                all_points_x += head_point_x * P[0]
                all_points_y += head_point_y * P[0]
                all_points_score += head_point_score * P[0]

                for j in range(5, 17):
                    all_points_x += elem["keypoints"][3*j]*P[j-4]
                    all_points_y += elem["keypoints"][3*j+1]*P[j-4]
                    all_points_score += elem["keypoints"][3*j+2]*P[j-4]

            gravity_point_x = all_points_x
            gravity_point_y = all_points_y
            gravity_point_score = all_points_score
            # 移除肩部中点
            elem["keypoints"].pop()
            elem["keypoints"].pop()
            elem["keypoints"].pop()
            # 添加重心
            elem["keypoints"].append(gravity_point_x)
            elem["keypoints"].append(gravity_point_y)
            elem["keypoints"].append(gravity_point_score)

            # 计算肩部的中点
            # should_mid_x = (elem["keypoints"][3 * 5 + 0] +
            #                 elem["keypoints"][3 * 6 + 0]) / 2
            # should_mid_y = (elem["keypoints"][3 * 5 + 1] +
            #                 elem["keypoints"][3 * 6 + 1]) / 2
            # should_mid_score = (
            #     elem["keypoints"][3 * 5 + 2] + elem["keypoints"][3 * 6 + 2]) / 2
            # elem["keypoints"].append(should_mid_x)
            # elem["keypoints"].append(should_mid_y)
            # elem["keypoints"].append(should_mid_score)
        with open(json_file_abs_path, 'w') as f:
            json.dump(keypoints_list, f)
        print("finished file:{}".format(json_file))
        print("************************************************************************************************")

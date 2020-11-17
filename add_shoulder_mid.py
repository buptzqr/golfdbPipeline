import os
import json
# 给已经处理好关键点文件添加肩部的中点（一个新关键点）
# total_keypoints_path = "/home/zqr/data/test/total_keypoints"
total_keypoints_path = "/home/zqr/data/golfdb_keypoints/all_keypoints"

if __name__ == '__main__':
    for json_file in os.listdir(total_keypoints_path):
        print("process file:{}".format(json_file))
        json_file_abs_path = os.path.join(total_keypoints_path, json_file)
        with open(json_file_abs_path, 'r') as f:
            keypoints_list = json.load(f)
        for elem in keypoints_list:
            should_mid_x = (elem["keypoints"][3 * 5 + 0] +
                            elem["keypoints"][3 * 6 + 0]) / 2
            should_mid_y = (elem["keypoints"][3 * 5 + 1] +
                            elem["keypoints"][3 * 6 + 1]) / 2
            should_mid_score = (
                elem["keypoints"][3 * 5 + 2] + elem["keypoints"][3 * 6 + 2]) / 2
            elem["keypoints"].append(should_mid_x)
            elem["keypoints"].append(should_mid_y)
            elem["keypoints"].append(should_mid_score)
        with open(json_file_abs_path, 'w') as f:
            json.dump(keypoints_list, f)
        print("finished file:{}".format(json_file))
        print("************************************************************************************************")

import os
import json
# 这个脚本是为了整合人体关键点数据和球杆关键点数据
if __name__ == '__main__':
    human_keypoints_path = "/home/zqr/data/golfdb_keypoints/human_keypoints/result"
    golf_club_keypoints_path = "/home/zqr/data/golfdb_keypoints/club_keypoints/result"
    total_keypoints_path = "/home/zqr/data/golfdb_keypoints/all_keypoints"

    # test时用
    # human_keypoints_path = "/home/zqr/data/test/test_keypoints/keypoints_result"
    # golf_club_keypoints_path = "/home/zqr/data/test/test_club_keypoints"
    # total_keypoints_path = "/home/zqr/data/test/total_keypoints"
    for folder in os.listdir(human_keypoints_path):
        human_kp_abs_path = os.path.join(
            human_keypoints_path, folder, "results.json")
        golf_club_kp_abs_path = os.path.join(
            golf_club_keypoints_path, folder, "club_keypoints.json")
        with open(human_kp_abs_path, "r") as f:
            human_list = json.load(f)
        with open(golf_club_kp_abs_path, "r") as f:
            club_list = json.load(f)
        human_list.sort(key=lambda res: res["image_id"])
        club_list.sort(key=lambda res: res["image_id"])
        # 如果不存在就取左右手肘中点和左右手腕中点的延长线，长度为40和200作为golf球杆的杆尾和杆头(因为要做st-gcn normalize应该绝对长度影响不大)
        # （当然必须得是在图片范围内的）
        for elem in human_list:
            elbow_first_index = 21
            elbow_second_index = 24
            wrist_first_index = 27
            wrist_second_index = 30

            elbow_mid_x = (elem["keypoints"][elbow_first_index] +
                           elem["keypoints"][elbow_second_index]) / 2
            elbow_mid_y = (elem["keypoints"][elbow_first_index+1] +
                           elem["keypoints"][elbow_second_index+1]) / 2
            elbow_mid_score = (
                elem["keypoints"][elbow_first_index+2] + elem["keypoints"][elbow_first_index+2]) / 2

            wrist_mid_x = (elem["keypoints"][wrist_first_index] +
                           elem["keypoints"][wrist_second_index]) / 2
            wrist_mid_y = (elem["keypoints"][wrist_first_index+1] +
                           elem["keypoints"][wrist_second_index+1]) / 2
            wrist_mid_score = (elem["keypoints"][wrist_first_index + 2] +
                               elem["keypoints"][wrist_second_index + 2]) / 2
            club_score = (elbow_mid_score + wrist_mid_score)/2
            if abs(elbow_mid_x - wrist_mid_x) < 1e-4:
                club_head_x = wrist_mid_x
                club_head_y = wrist_mid_y + 200
                club_tail_x = wrist_mid_x
                club_tail_y = wrist_mid_y + 40
            else:
                slop = (elbow_mid_y - wrist_mid_y) / \
                    (elbow_mid_x - wrist_mid_x)
                club_head_x = wrist_mid_x + 200 * slop
                club_head_y = wrist_mid_y + 200 * slop
                club_tail_x = wrist_mid_x + 40 * slop
                club_tail_y = wrist_mid_x + 40 * slop
            elem["keypoints"].append(club_head_x)
            elem["keypoints"].append(club_head_y)
            elem["keypoints"].append(club_score)
            elem["keypoints"].append(club_tail_x)
            elem["keypoints"].append(club_tail_y)
            elem["keypoints"].append(club_score)
        for e_c in club_list:
            img_id = e_c["image_id"]
            keypoints = []
            for k in e_c["keypoints"]:
                keypoints.extend(k)
            for e_h in human_list:
                if e_h["image_id"] == img_id:
                    for i in range(6):
                        e_h["keypoints"][51+i] = keypoints[0+i]

        print("process folder:{}".format(folder))
        with open(os.path.join(total_keypoints_path, folder + ".json"), "w") as f:
            json.dump(human_list, f)
        print("finish folder:{}".format(folder))
        print("********************************")

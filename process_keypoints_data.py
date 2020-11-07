import os
import json
# 这个脚本是为了整合人体关键点数据和球杆关键点数据
if __name__ == '__main__':
    human_keypoints_path = ""
    golf_club_keypoints_path = ""
    total_keypoints_path = ""
    for folder in os.listdir(human_keypoints_path):
        human_kp_abs_path = os.path.join(
            human_keypoints_path, folder, "results.json")
        golf_club_kp_abs_path = os.path.join(
            golf_club_keypoints_path, folder, "results.json")
        with open(human_kp_abs_path, "r") as f:
            human_list = json.load(f)
        with open(golf_club_keypoints_path, "r") as f:
            club_list = json.load(f)
        human_list.sort(key=lambda res: res["image_id"])
        club_list.sort(key=lambda res: res["image_id"])
        assert len(human_list) == len(club_list)
        for i in range(len(human_list)):
            if human_list[i]["image_id"] == club_list[i]["image_id"]:
                for elem_list in club_list[i]["keypoints"]:
                    for elem in elem_list:
                        human_list[i]["keypoints"].append(elem)
        with open(os.path.join(total_keypoints_path, folder + ".json"), "w") as f:
            json.dump(human_list, f)

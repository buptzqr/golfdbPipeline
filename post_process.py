# 这个脚本主要用来进行后处理操作
from data.config import cfg
import json
import os
if __name__ == '__main__':
    ANGLE = 'FRONT' # 分为两种，正面(FRONT)和侧面(SIDE)
    for res_dir in os.listdir(cfg.TEST_RESULT_PATH):
        file_name_list =[]
        for file_name in os.listdir(os.path.join(cfg.TEST_RESULT_PATH, res_dir)):
            file_name_list.append(file_name)
        #排序
        file_name_list.sort()
        sequence_list = []
        for e in file_name_list:
            sequence_list.append(e.split('.')[0].split('_')[1])

        # 6（下杆手臂水平），9（击球），11（送杆杆身水平）这三帧准确率比较高，以delta为搜索范围进一步矫正这三帧
        # 使用关键点信息来进行矫正
        DELTA = 4
        anchor_frame_num_list = [5,10]
        
        for e in anchor_frame_num_list:
            if e == 5:
                # 这帧是手臂水平，需要使用人体关键点信息来进一步矫正
                human_keypoints_path = os.path.join(cfg.TEST_KEYPOINTS_PATH,'keypoints_result' ,res_dir, 'results.json')
                with open(human_keypoints_path,'r') as f:
                    human_keypoints_json = json.load(f)
                    human_keypoints_json.sort(key= lambda res: res["image_id"])
                    slop_5 = 1
                    frame_5th_idx = int(sequence_list[e])
                    for search_idx in range(int(sequence_list[e]) - DELTA, int(sequence_list[e]) + DELTA + 1):
                        # 左肘是8，左腕是10，右肘是9，右腕是11
                        left_elbow_x = human_keypoints_json[search_idx]['keypoints'][7 * 3]
                        left_elbow_y = human_keypoints_json[search_idx]['keypoints'][7 * 3 + 1]
                        left_wrist_x = human_keypoints_json[search_idx]['keypoints'][9 * 3]
                        left_wrist_y = human_keypoints_json[search_idx]['keypoints'][9 * 3 + 1]
                        if abs(left_elbow_x-left_wrist_x)<1e-4:
                            continue
                        left_slop = abs((left_wrist_y-left_elbow_y)/(left_elbow_x-left_wrist_x))
                        if left_slop < slop_5:
                            slop_5 = left_slop
                            frame_5th_idx = search_idx
            else:
                club_keypoints_path = os.path.join(cfg.TEST_CLUB_KEYPOINTS_PATH, res_dir, 'club_keypoints.json')
                with open(club_keypoints_path, 'r') as f:
                    club_keypoints_json = json.load(f)
                    club_keypoints_json.sort(key= lambda res: res["image_id"])
                    slop_10 = 1                   
                    frame_10th_idx = int(sequence_list[e]) 
                    for search_idx in range(int(sequence_list[e]) - DELTA, int(sequence_list[e]) + DELTA + 1):
                        club_head_x = club_keypoints_json[search_idx]['keypoints'][0][0]
                        club_head_y = club_keypoints_json[search_idx]['keypoints'][0][1]
                        club_tail_x = club_keypoints_json[search_idx]['keypoints'][1][0]
                        club_tail_y = club_keypoints_json[search_idx]['keypoints'][1][1]
                        if abs(club_tail_x-club_head_x) < 1e-4:
                            continue
                        club_slop = abs((club_head_y-club_tail_y)/(club_tail_x-club_head_x))
                        if club_slop < slop_10:
                            slop_10 = club_slop  
                            frame_10th_idx = search_idx
            
        # 寻找下杆杆身水平和下杆杆身45 
        impact_delta = 2
        frame_6th_idx = int(sequence_list[6])
        frame_7th_idx = int(sequence_list[7])
        slop6 = 1
        slop7 = 4
        with open(club_keypoints_path, 'r') as f:
            for search_idx in range(frame_5th_idx,int(sequence_list[8])+impact_delta):
                club_head_x = club_keypoints_json[search_idx]['keypoints'][0][0]
                club_head_y = club_keypoints_json[search_idx]['keypoints'][0][1]
                club_tail_x = club_keypoints_json[search_idx]['keypoints'][1][0]
                club_tail_y = club_keypoints_json[search_idx]['keypoints'][1][1]
                if abs(club_tail_x-club_head_x) < 1e-4:
                    continue
                club_slop = abs((club_head_y-club_tail_y)/(club_tail_x-club_head_x))
                if club_slop < slop6:                    
                    slop6 = club_slop
                    frame_6th_idx = search_idx
                if abs(club_slop-1) < abs(slop7-1):
                    slop7 = club_slop
                    frame_7th_idx = search_idx
        # 寻找送杆杆身45
        frame_9th_idx = int(sequence_list[9])
        slop9 = 4
        with open(club_keypoints_path, 'r') as f:
            for search_idx in range(int(sequence_list[8])-impact_delta,frame_10th_idx):
                club_head_x = club_keypoints_json[search_idx]['keypoints'][0][0]
                club_head_y = club_keypoints_json[search_idx]['keypoints'][0][1]
                club_tail_x = club_keypoints_json[search_idx]['keypoints'][1][0]
                club_tail_y = club_keypoints_json[search_idx]['keypoints'][1][1]
                if abs(club_tail_x-club_head_x) < 1e-4:
                    continue
                club_slop = abs((club_head_y-club_tail_y)/(club_tail_x-club_head_x))
                if abs(club_slop -1) < abs(slop9-1):
                    slop9 = club_slop
                    frame_9th_idx = search_idx
        
        print("5th index:{}".format(frame_5th_idx))
        print("6th index:{}".format(frame_6th_idx))
        print("7th index:{}".format(frame_7th_idx))
        print("8th index:{}".format(int(sequence_list[8])))
        print("9th index:{}".format(frame_9th_idx))
        print("10th index:{}".format(frame_10th_idx))

                                
# 8帧13帧融合以后的准确率评价
# 使用该脚本要更改displaydataloader中的48行
# for img_name in range(1, img_num+1):
import os
from data.config import cfg
import json
import numpy as np
import pandas as pd
import sys
import shutil

if __name__ == "__main__":
    # if len(sys.argv) != 2:
    #     print("you should give model split")
    #     sys.exit(1)
    # split = sys.argv[1]
    split = 4
    json_dir = cfg.VAL_JSON_PATH
    json_files = []
    for file in os.listdir(json_dir):
        json_files.append(file)
    df = pd.read_pickle(os.path.join("./data","val_split_{}.pkl".format(split)))
    correct = []
    cnt = 0
    for file in json_files:
        cnt = cnt + 1
        ground_truth_list = []
        merge_list = []
        with open(os.path.join(json_dir, file), 'r') as f:
            video_name = file[:11]
            json_str = f.read()
            json_data = json.loads(json_str)
            json_id = int(file[12])
            videos_info = df[df["youtube_id"] == video_name]
            if videos_info.empty:
                continue
            # 得到了id，也就是范围[0,1400)的数字,根据这个id去做好光流图和切好原始图的文件夹中读取数据
            video_id = str(videos_info.iloc[json_id]["id"])
            print(video_id)
            # 将视频复制到test_video，将光流文件复制到test_opt,将图片复制到test_img
            video_src_path = os.path.join("/home/zqr/data/golfdb_split_no_resize","{}.mp4".format(video_id))
            video_dst_path = os.path.join(cfg.TEST_VIDEO_PAHT,"{}.mp4".format(video_id))
            shutil.copyfile(video_src_path, video_dst_path)
            
            opt_src_path = os.path.join(cfg.OPT_RESIZE_FILE_PATH,str(video_id))
            opt_dst_path = os.path.join(cfg.TEST_OPT_DIR,str(video_id))
            if os.path.exists(opt_dst_path):
                shutil.rmtree(opt_dst_path)
            shutil.copytree(opt_src_path, opt_dst_path, True)

            img_src_path = os.path.join(cfg.IMG_FRAME_PATH,str(video_id))
            img_dst_path = os.path.join(cfg.TEST_IMGS_DIR,str(video_id))
            if os.path.exists(img_dst_path):
                shutil.rmtree(img_dst_path)
            shutil.copytree(img_src_path, img_dst_path, True)
            
            # 得到了对应的帧号
            ground_truth_list = np.array(json_data["categories"][0]["events"])
            # 执行8帧和13帧融合
            # 8帧
            runStatus = os.system(
                'python3 disply.py {}'.format(True))
            if runStatus != 0:
                print("关键帧8帧提取错误")
                sys.exit(5)
            print("关键帧8帧提取完成")
            
            # 13帧
            runStatus = os.system(
                'python3 disply.py {}'.format(False))
            if runStatus != 0:
                print("关键帧13帧提取错误")
                sys.exit(5)
            print("关键帧13帧提取完成")
            
            runStatus = os.system('python3 merge_result.py')
            if runStatus != 0:
                print("merge错误")
                sys.exit(5)
            print("merge完成")
            #融合后结果在result中
            merge_result_dir = os.path.join(cfg.TEST_RESULT_PATH,video_id)
            for frame_num in os.listdir(merge_result_dir):
                merge_list.append(frame_num)
            
        #计算准确率
        merge_list.sort()
        for i in range(len(merge_list)):
            merge_list[i] = int(merge_list[i].split('_')[1].split('.')[0])
        merge_list = np.array(merge_list)
        ground_truth_list = np.array(ground_truth_list)
        tol = int(max(np.round((ground_truth_list[8] - ground_truth_list[0]) / 30), 1))
        deltas = np.abs(merge_list-ground_truth_list)
        c = (deltas <= tol).astype(np.uint8)
        correct.append(c)
    print("cnt:{}".format(cnt))    
    PFCR = np.mean(correct,axis=0)
    PCE = np.mean(correct)
    print(PFCR)
    print(PCE)
            

            
    
        

    
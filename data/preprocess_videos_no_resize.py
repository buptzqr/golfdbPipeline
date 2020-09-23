import pandas as pd
import os
import cv2
from multiprocessing import Pool
import numpy as np
import json
from config import cfg
# 本文件用来从580高尔夫视频中切割出golfdb对应的1400文件，同时记录bbox方便后续处理
# box文件存储的格式是-filename:[x,y,w,h]
df = pd.read_pickle('./golfDB.pkl')
yt_video_dir = cfg.YOUTUBE_PATH
bbox_path = cfg.BBOX_INFO_PATH
video_count = 0


def preprocess_videos(anno_id):
    """
    Extracts relevant frames from youtube videos
    """
    global video_count
    a = df.loc[df['id'] == anno_id]
    path = cfg.NEW_VIDEO_PATH

    youtube_id = a['youtube_id'][anno_id]
    events = a['events'][anno_id]
    bbox = a['bbox'][anno_id]
    with open(bbox_path, 'a') as f:
        f.write(str(anno_id))
        f.write(':')

        if not os.path.isfile(os.path.join(path, '{}.mp4'.format(str(video_count)))):
            print('Processing annotation id {}'.format(anno_id))
            cap = cv2.VideoCapture(os.path.join(
                yt_video_dir, '{}.mp4'.format(youtube_id)))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            x = int(width * bbox[0])
            y = int(height * bbox[1])
            w = int(width * bbox[2])
            h = int(height * bbox[3])
            box_list = [x, y, w, h]
            f.write(str(box_list))
            f.write('\n')
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(os.path.join(path, '{}.mp4'.format(str(video_count))), fourcc, cap.get(cv2.CAP_PROP_FPS),
                                  (width, height))

            count = 0
            success, image = cap.read()
            while success:
                count += 1
                if count >= events[0] and count <= events[-1]:
                    out.write(image)
                if count > events[-1]:
                    break
                success, image = cap.read()
        video_count += 1


if __name__ == '__main__':
    path = cfg.NEW_VIDEO_PATH

    if not os.path.exists(path):
        os.mkdir(path)
    if os.path.exists(bbox_path):
        os.remove(bbox_path)
    for i in range(len(df.id)):
        preprocess_videos(df.id[i])

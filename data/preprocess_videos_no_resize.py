import pandas as pd
import os
import cv2
from multiprocessing import Pool
import numpy as np
import json

df = pd.read_pickle('./golfDB.pkl')
yt_video_dir = '/home/zqr/codes/data/golfdb_correct'
video_count = 0


def preprocess_videos(anno_id):
    """
    Extracts relevant frames from youtube videos
    """
    global video_count
    a = df.loc[df['id'] == anno_id]
    path = '/home/zqr/codes/data/golfdb_split_no_resize'

    youtube_id = a['youtube_id'][anno_id]
    events = a['events'][anno_id]

    if not os.path.isfile(os.path.join(path, '{}.mp4'.format(str(video_count)))):
        print('Processing annotation id {}'.format(anno_id))
        cap = cv2.VideoCapture(os.path.join(
            yt_video_dir, '{}.mp4'.format(youtube_id)))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
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
    path = '/home/zqr/codes/data/golfdb_split_no_resize'

    if not os.path.exists(path):
        os.mkdir(path)

    for i in range(len(df.id)):
        preprocess_videos(df.id[i])

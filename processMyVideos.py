import pandas as pd
import os
import cv2
from multiprocessing import Pool
import numpy as np
import sys

videosFolder = ''
videosBboxPath = ''

def preprocess_videos(imagesFolder_160, dim=160):
    bboxMap = {}
    with open(videosBboxPath) as f:
        for line in f.readlines():
            bbox = []
            line = line.rstrip()
            bboxKey = line.split(' ')[0]
            tmp = line.split(' ',1)[1]
            tmp = tmp.rstrip(']')
            tmp = tmp.strip('[')
            a1= tmp.split(',',3)[0]
            a2= tmp.split(',',3)[1]
            a3= tmp.split(',',3)[2]
            a4= tmp.split(',',3)[3]
            bbox.append(int(a1))
            bbox.append(int(a2))
            bbox.append(int(a3))
            bbox.append(int(a4))
            bboxMap[bboxKey]=bbox

    for videoFile in os.listdir(videosFolder):
        videoName = videoFile.split('.')[0]
        bbox = bboxMap[videoName]
        
        path = imagesFolder_160
        if not os.path.exists(path):
            os.mkdir(path)
        
        if not os.path.isfile(os.path.join(path, "{}.mp4".format(videoName))):
            print('Processing annotation id {}'.format(videoName))
            cap = cv2.VideoCapture(os.path.join(videosFolder, '{}.mp4'.format(videoName)))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            # out = cv2.VideoWriter(os.path.join(path, "{}.mp4".format(videoName)),
            #                     fourcc, cap.get(cv2.CAP_PROP_FPS), (dim, dim))
            imagesFoler = os.path.join(path, videoName)
            if not os.path.exists(imagesFoler):
                os.mkdir(imagesFoler)
            x = int (bbox[0])
            y = int (bbox[1])
            w = int (bbox[2])
            h = int (bbox[3])
            
            count = 0
            success, image = cap.read()
            imageNum = 0
            while success:
                count += 1
                # if count >= events[0] and count <= events[-1]:
                crop_img = image[y:y + h, x:x + w]
                crop_size = crop_img.shape[:2]
                ratio = dim / max(crop_size)
                new_size = tuple([int(x*ratio) for x in crop_size])
                resized = cv2.resize(crop_img, (new_size[1], new_size[0]))
                delta_w = dim - new_size[1]
                delta_h = dim - new_size[0]
                top, bottom = delta_h // 2, delta_h - (delta_h // 2)
                left, right = delta_w // 2, delta_w - (delta_w // 2)
                b_img = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                            value=[0.406*255, 0.456*255, 0.485*255])  # ImageNet means (BGR)
                cv2.imwrite(os.path.join(imagesFoler,'{:0>4d}.jpg'.format(imageNum)),b_img)
                # out.write(b_img)
                # if count > events[-1]:
                #     break
                imageNum += 1
                success, image = cap.read()
            print("resize {} files".format(count))
        else:
            print('video file {} already completed for size {}'.format(videoName, dim))


if __name__ == '__main__':
    # if len(sys.argv) != 4:
    #     print("not enough param in processMyVideos")
    #     sys.exit(1)
    # videosBboxPath = sys.argv[1]
    # imagesFolder_160 = sys.argv[2]
    # videosFolder = sys.argv[3]
    
    videosBboxPath = "/home/zqr/codes/data/videosBbox.txt"
    imagesFolder_160 = "/home/zqr/codes/data/imagesFolder_160"
    videosFolder = "/home/zqr/codes/data/golfVideos"

    preprocess_videos(imagesFolder_160)

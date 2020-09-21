import pandas as pd
import os
import cv2
from multiprocessing import Pool
import numpy as np
import sys
import numpy

optFilesFolder = ''
videosBboxPath = ''


def preprocess_optFiles(optFilesFolder_160, dim=160):
    bboxMap = {}
    with open(videosBboxPath) as f:
        for line in f.readlines():
            bbox = []
            line = line.rstrip()
            bboxKey = line.split(' ')[0]
            tmp = line.split(' ', 1)[1]
            tmp = tmp.rstrip(']')
            tmp = tmp.strip('[')
            a1 = tmp.split(',', 3)[0]
            a2 = tmp.split(',', 3)[1]
            a3 = tmp.split(',', 3)[2]
            a4 = tmp.split(',', 3)[3]
            bbox.append(int(a1))
            bbox.append(int(a2))
            bbox.append(int(a3))
            bbox.append(int(a4))
            bboxMap[bboxKey] = bbox

    count = 0  # 记录处理了多少文件夹
    for optFolder in os.listdir(optFilesFolder):
        bbox = bboxMap[optFolder]

        path = optFilesFolder_160
        if not os.path.exists(path):
            os.mkdir(path)

        print('Processing folder id {}'.format(optFolder))

        optFilesResizeFoler = os.path.join(path, optFolder)
        if not os.path.exists(optFilesResizeFoler):
            os.mkdir(optFilesResizeFoler)
        x = int(bbox[0])
        y = int(bbox[1])
        w = int(bbox[2])
        h = int(bbox[3])
        folderPath = os.path.join(optFilesFolder, optFolder)
        fileNum = 0  # 记录resize了多少文件
        for optFile in os.listdir(folderPath):
            # 这个图片尺寸需要根据需要更改
            filePath = os.path.join(folderPath, optFile)
            opticalOri = np.fromfile(
                filePath, np.float32, offset=12).reshape(960, 544, 2)
            opticalArray = np.empty([960, 544, 3], np.float32)
            opticalArray[..., 0] = 255
            opticalArray[..., 1] = opticalOri[:, :, 0]
            opticalArray[..., 2] = opticalOri[:, :, 1]

            # if count >= events[0] and count <= events[-1]:
            crop_img = opticalArray[y:y + h, x:x + w]
            crop_size = crop_img.shape[:2]
            ratio = dim / max(crop_size)
            new_size = tuple([int(x*ratio) for x in crop_size])
            resized = cv2.resize(crop_img, (new_size[1], new_size[0]))
            delta_w = dim - new_size[1]
            delta_h = dim - new_size[0]
            top, bottom = delta_h // 2, delta_h - (delta_h // 2)
            left, right = delta_w // 2, delta_w - (delta_w // 2)
            # b_img = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT,
            #                            value=[0.406*255, 0.456*255, 0.485*255])  # ImageNet means (BGR)

            b_img = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                       value=[0, 0, 0])  # ImageNet means (BGR)

            # b_img = cv2.copyMakeBorder(
            #     resized, top, bottom, left, right, cv2.BORDER_REPLICATE)

            opt_160_file_path = os.path.join(optFilesResizeFoler, optFile)
            objOutput = open(opt_160_file_path, 'wb')

            numpy.array([80, 73, 69, 72], numpy.uint8).tofile(objOutput)
            opticalArray = np.empty([160, 160, 2], np.float32)
            opticalArray[..., 0] = b_img[..., 1]
            opticalArray[..., 1] = b_img[..., 2]

            numpy.array([opticalArray.shape[2], opticalArray.shape[1]],
                        numpy.int32).tofile(objOutput)
            numpy.array(opticalArray, numpy.float32).tofile(objOutput)

            objOutput.close()
            fileNum += 1
        count += 1
        print("resize {} files".format(fileNum))


if __name__ == '__main__':
    # optFilesFolder原始光流图所在位置

    # if len(sys.argv) != 4:
    #     print("not enough param in optFileProcess")
    #     sys.exit(1)
    # videosBboxPath = sys.argv[1]
    # optFilesFolder_160 = sys.argv[2]
    # optFilesFolder = sys.argv[3]

    videosBboxPath = "/home/zqr/codes/data/videosBbox.txt"
    optFilesFolder_160 = "/home/zqr/codes/data/optFromOri_160"
    optFilesFolder = "/home/zqr/codes/data/optOri"

    preprocess_optFiles(optFilesFolder_160)

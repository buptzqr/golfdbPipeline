import sys
import os
if __name__ == '__main__':
    # 用来整理整个流程
    # videosFolder 存放视频数据位置
    # imagesFolder 视频帧存放位置
    # opticalFolder 存放光流文件位置
    # dataInfoPath 数据文件存放位置
    # videosBboxPath 存放视频的bbox
    # imagesFolder_160 存放裁剪成160×160的视频帧
    # resultFolder 存放关键帧结果
    # optFilesFolder 存放原始视频光流图
    # optFilesFolder_160 存放resize成160×160光流图
    # if len(sys.argv) != 6:
    #     print("you should give enough param")
    #     sys.exit(1)
    # videosFolder = sys.argv[1]
    # imagesFolder = sys.argv[2]
    # opticalFolder = sys.argv[3]
    # dataInfoPath = sys.argv[4]
    # videosBboxPath = sys.argv[5]
    # imagesFolder_160 = sys.argv[6]
    # resultFolder = sys.argv[7]
    videosFolder = "/home/zqr/codes/data/golfdb_split_no_resize"
    imagesFolder = "/home/zqr/codes/data/glofdb_frame_no_resize"
    opticalFolder = "/home/zqr/codes/data/opticalFiles"
    dataInfoPath = "/home/zqr/codes/data/data_info.txt"
    videosBboxPath = "/home/zqr/codes/data/videosBbox.txt"
    imagesFolder_160 = "/home/zqr/codes/data/glofdb_frame_no_resize"
    resultFolder = "/home/zqr/codes/data/resultFolder"
    optFilesFolder_160 = "/home/zqr/codes/data/optFromOri_160"
    optFilesFolder = "/home/zqr/codes/data/optOri"

    # # step1: 切帧
    runStatus = os.system(
        'python3 getFrames.py {} {}'.format(videosFolder, imagesFolder))
    if runStatus != 0:
        print("切帧流程错误")
        sys.exit(1)
    print("切帧流程完成")

    # # step2：MaskRCNN 检测bouldingbox
    # os.chdir("/home/zqr/codes/Mask_RCNN-master")
    # runStatus = os.system('python3 getBbox.py {} {}'.format(
    #     videosBboxPath, imagesFolder))
    # if runStatus != 0:
    #     print("检测候选框流程错误")
    #     sys.exit(1)
    # print("检测候选框流程完成")

    # # step3: 根据bbox 将图片重构为160×160
    # os.chdir("/home/zqr/codes/MyGolfDB")
    # runStatus = os.system('python3 processMyVideos.py {} {} {}'.format(
    #     videosBboxPath, imagesFolder_160, videosFolder))
    # if runStatus != 0:
    #     print("重构160流程错误")
    #     sys.exit(1)
    # print("重构160流程完成")

    # # step4: 光流法提取特征
    # os.chdir("/home/zqr/codes/pytorch-liteflownet")
    # runStatus = os.system('python3 run.py {} {} {}'.format(
    #     imagesFolder_160, opticalFolder, dataInfoPath))
    # if runStatus != 0:
    #     print("光流法提取特征错误")
    #     sys.exit(1)
    # print("光流法提取特征完成")

    # # step4*：将原始光流图重构为160×160
    # os.chdir("/home/zqr/codes/MyGolfDB")
    # runStatus = os.system(
    #     'python3 optFileProcess.py {} {} {}'.format(videosBboxPath, optFilesFolder_160, optFilesFolder))
    # if runStatus != 0:
    #     print("将原始光流图重构为160×160错误")
    #     sys.exit(1)
    # print("将原始光流图重构为160×160完成")

    # # step5: 提取关键帧
    # os.chdir("/home/zqr/codes/MyGolfDB")
    # runStatus = os.system('python3 disply.py {} {}'.format(dataInfoPath, resultFolder))
    # if runStatus != 0:
    #     print("关键帧提取错误")
    #     sys.exit(1)
    # print("关键帧提取完成")

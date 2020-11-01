import sys
import os
from data.config import cfg
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
    videosFolder = "/home/zqr/data/test/test_video"
    imagesFolder = "/home/zqr/data/test/test_img"
    opticalFolder = "/home/zqr/codes/data/opticalFiles"
    dataInfoPath = "/home/zqr/codes/data/data_info.txt"
    videosBboxPath = "/home/zqr/data/test/videosBbox.txt"
    imagesFolder_160 = "/home/zqr/codes/data/glofdb_frame_no_resize"
    resultFolder = "/home/zqr/codes/data/resultFolder"
    optFilesFolder_160 = "/home/zqr/codes/data/optFromOri_160"
    optFilesFolder = "/home/zqr/codes/data/optOri"

    # # step1: 切帧
    # runStatus = os.system(
    #     'python3 getFrames.py')
    # if runStatus != 0:
    #     print("切帧流程错误")
    #     sys.exit(1)
    # print("切帧流程完成")

    # # step2：FPN 检测bouldingbox
    # runStatus = os.system('python3 getInfo.py')
    # if runStatus != 0:
    #     print("检测候选框流程错误")
    #     sys.exit(2)
    # print("检测候选框流程完成")

    # # # step3: 根据bbox 将图片重构为dim×dim(这一步待定吧)
    # os.chdir("/home/zqr/codes/MyGolfDB")
    # runStatus = os.system('python3 resizeFrame.py')
    # if runStatus != 0:
    #     print("resize流程错误")
    #     sys.exit(3)
    # print("resize流程完成")

    # # step4: 光流法提取特征(flag是用来打开resize开关的，dim需要在光流proj的config里面配置一下，这样做的话就是在原图提光流特征，然后resize)
    # 这里没有完应该是根据bbox resize比较好吧
    os.chdir("/home/zqr/codes/pytorch-liteflownet")
    runStatus = os.system('python3 run.py {} {} {} {}'.format(
        cfg.TEST_IMGS_DIR, cfg.TEST_OPT_DIR, cfg.OPT_RESIZE_FLAG, cfg.TEST_BBOX_INFO_PATH))
    if runStatus != 0:
        print("光流法提取特征错误")
        sys.exit(1)
    print("光流法提取特征完成")

    # # step5: 提取关键帧
    os.chdir("/home/zqr/codes/MyGolfDB")
    runStatus = os.system(
        'python3 disply.py {} {}'.format(dataInfoPath, resultFolder))
    if runStatus != 0:
        print("关键帧提取错误")
        sys.exit(1)
    print("关键帧提取完成")

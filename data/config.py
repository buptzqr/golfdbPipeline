from enum import Enum


class Config:
    NEW_VIDEO_PATH = "/home/zqr/data/golfdb_split_no_resize"
    YOUTUBE_PATH = "/home/zqr/data/golfdb_correct"
    OPT_ORI_FILE_PATH = "/home/zqr/data/golfdb_flow_no_resize"
    OPT_RESIZE_FILE_PATH = "/home/zqr/data/optical/opticalFlowOri_160"
    VIDEO_160_PATH = "/home/zqr/data/videos_160"
    TRAIN_JSON_PATH = "/home/zqr/data/golfdb_json/train"
    VAL_JSON_PATH = "/home/zqr/data/golfdb_json/val"
    BBOX_CLUB_POINTS_MODEL_PATH = "/home/zqr/codes/MyGolfDB/model_final.pth"
    INPUT_DIM = 160
    SPLIT = 4  # 保证eval和train是同一个split
    ITERATIONS = 500
    IT_SAVE = 100
    SEQUENCE_LENGTH = 64
    CPU_NUM = 6
    BATCH_SIZE = 20
    PKL_FILE_PATH = "./data/train_split_{}.pkl".format(SPLIT)
    OUR_PKL_FILE_PATH = "./data/golfDB.pkl"
    FRAME_13_OPEN = True
    #############################################################################
    # test config #
    RESIZE_DIM = 360
    # 存放每一帧图片bbox(球杆和人)的位置
    TEST_BBOX_INFO_PATH = "/home/zqr/data/test/test_bbox"
    TEST_IMGS_DIR = "/home/zqr/data/test/test_img"  # 存放切好的视频帧的位置
    TEST_OPT_DIR = "/home/zqr/data/test/test_opt"  # 存放提取好光流信息的视频帧
    TEST_VIDEO_PAHT = "/home/zqr/data/test/test_video"  # 存放待检测的视频帧
    # 存放提取好的关键点信息(包括球杆和人)的位置
    TEST_KEYPOINTS_PATH = "/home/zqr/data/test/test_keypoints"
    TEST_CLUB_KEYPOINTS_PATH = "/home/zqr/data/test/test_club_keypoints"  # 存放提取好的球杆关键点
    OPT_RESIZE_FLAG = True
    TEST_RESULT_PATH = "/home/zqr/data/test/result"
    TEST_MODEL = "./models/swingnet_100.pth.tar"

    class DATALOADER_OPT(Enum):
        RGB = 0
        OPTICAL_FLOW = 1
        KEYPOINTS = 2

    DATAOPT = DATALOADER_OPT.OPTICAL_FLOW


config = Config()
cfg = config

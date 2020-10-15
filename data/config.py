from enum import Enum


class Config:
    NEW_VIDEO_PATH = "/home/zqr/data/golfdb_split_no_resize"
    BBOX_INFO_PATH = "/home/zqr/data/bbox.txt"
    YOUTUBE_PATH = "/home/zqr/data/golfdb_correct"
    OPT_ORI_FILE_PATH = "/home/zqr/data/golfdb_flow_no_resize"
    OPT_RESIZE_FILE_PATH = "/home/zqr/data/optical/opticalFlowOri_160"
    VIDEO_160_PATH = "/home/zqr/data/videos_160"
    TRAIN_JSON_PATH = "/home/zqr/data/golfdb_json/train"
    VAL_JSON_PATH = "/home/zqr/data/golfdb_json/val"
    INPUT_DIM = 160
    SPLIT = 4  # 保证eval和train是同一个split
    ITERATIONS = 2500
    IT_SAVE = 100
    SEQUENCE_LENGTH = 64
    CPU_NUM = 6
    BATCH_SIZE = 20
    PKL_FILE_PATH = "./data/train_split_{}.pkl".format(SPLIT)
    OUR_PKL_FILE_PATH = "./data/golfDB.pkl"
    FRAME_13_OPEN = True

    class DATALOADER_OPT(Enum):
        RGB = 0
        OPTICAL_FLOW = 1
        KEYPOINTS = 2

    DATAOPT = DATALOADER_OPT.OPTICAL_FLOW


config = Config()
cfg = config

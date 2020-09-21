import os
import cv2
import sys
if __name__=="__main__":
    # use to get frames
    if len(sys.argv) != 3:
        print("there is not enough param in getFrames")
        sys.exit(1)
    # videosFolder="/home/zqr/codes/GolfDB/data/videos_160"
    # imagesFolder="/home/zqr/codes/data/Videos2Frames_160"
    videosFolder = sys.argv[1]
    imagesFolder = sys.argv[2]  
    if not os.path.exists(imagesFolder):
            os.mkdir(imagesFolder)
    index=0
    for info in os.walk(videosFolder):
        fileNames = info
    for filename in fileNames[2]:
        process_file_dir = filename.split(".")[0]
        dst_dir=os.path.join(imagesFolder,str(process_file_dir))
        if not os.path.exists(dst_dir):
            os.mkdir(dst_dir)
        video = cv2.VideoCapture()
        if not video.open(os.path.join(videosFolder,filename)):
            print("can not open the video")
            exit(1)
        pic_index = 0
        while True:
            _, frame = video.read()
            if frame is None:
                break
            save_path = "{}/{:>04d}.jpg".format(dst_dir, pic_index)
            cv2.imwrite(save_path, frame)
            pic_index += 1 
        index += 1
        video.release()
        print("Totally process {:d} video".format(index))
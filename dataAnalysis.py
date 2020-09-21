import sys
import pandas as pd
# 主要进行数据分析
# 提取最后一帧和第一帧识别正确的video的特征
if __name__ == "__main__":
    split = sys.argv[1]
    val_summary_path = "./summary_{}.txt".format(split)
    val_info_path = "./data/val_split_{}_info.csv".format(split)
    summary = open(val_summary_path,'r')
    df_summary = pd.read_csv(val_summary_path)
    
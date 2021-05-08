# 这个脚本用来根据8帧和13帧检测的分数来确定最终的结果
import os
import json
from data.config import cfg
if __name__ == '__main__':
    scores_13 = {}
    scores_8 = {}
    with open(os.path.join(cfg.TEST_SCORES_DIR,"scores_8.json"),'r') as f:
        scores_8 = json.load(f)        
    with open(os.path.join(cfg.TEST_SCORES_DIR,"scores_13.json"),'r') as f:
        scores_13 = json.load(f)
    with open(os.path.join(cfg.TEST_SCORES_DIR,"preds_8.json"),'r') as f:
        preds_8 = json.load(f)        
    with open(os.path.join(cfg.TEST_SCORES_DIR,"preds_13.json"),'r') as f:
        preds_13 = json.load(f)
    
    for k,v in scores_8.items():
        dst_path = os.path.join(cfg.TEST_RESULT_PATH,str(k))
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)
        score_8 = scores_8[k]
        score_13 = scores_13[k]
        index_8_to_13 = {0:0,1:2,2:3,3:4,4:5,5:8,6:10,7:12}
        for idx_8,idx_13 in index_8_to_13.items():
            if score_8[idx_8]>score_13[idx_13]:
                src_path = os.path.join(cfg.TEST_RESULT_TMP,"tmp_result_8",str(k))
                src_img = os.path.join(src_path, "{:0>4d}_{}.jpg".format(idx_8,preds_8[k][idx_8]))
                dst_img = os.path.join(
                    dst_path, "{:0>4d}_{}.jpg".format(idx_13, preds_8[k][idx_8]))
                os.system("cp {} {}".format(src_img, dst_img))
            else:
                src_path = os.path.join(cfg.TEST_RESULT_TMP,"tmp_result_13",str(k))
                src_img = os.path.join(src_path, "{:0>4d}_{}.jpg".format(idx_13,preds_13[k][idx_13]))
                dst_img = os.path.join(
                    dst_path, "{:0>4d}_{}.jpg".format(idx_13, preds_13[k][idx_13]))
                os.system("cp {} {}".format(src_img, dst_img))
        other_idx = [1,6,7,9,11]
        for idx in other_idx:
            src_path = os.path.join(cfg.TEST_RESULT_TMP,"tmp_result_13",str(k))
            src_img = os.path.join(src_path, "{:0>4d}_{}.jpg".format(idx,preds_13[k][idx]))
            dst_img = os.path.join(
                dst_path, "{:0>4d}_{}.jpg".format(idx, preds_13[k][idx]))
            os.system("cp {} {}".format(src_img, dst_img))
            
        
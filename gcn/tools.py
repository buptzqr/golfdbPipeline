import numpy as np
import random
import json
import torch
import cv2
import os


def random_move(data_numpy,
                angle_candidate=[-10., -5., 0., 5., 10.],
                scale_candidate=[0.9, 1.0, 1.1],
                transform_candidate=[-0.2, -0.1, 0.0, 0.1, 0.2],
                move_time_candidate=[1]):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    move_time = random.choice(move_time_candidate)
    node = np.arange(0, T, T * 1.0 / move_time).round().astype(int)
    node = np.append(node, T)
    num_node = len(node)

    A = np.random.choice(angle_candidate, num_node)
    S = np.random.choice(scale_candidate, num_node)
    T_x = np.random.choice(transform_candidate, num_node)
    T_y = np.random.choice(transform_candidate, num_node)

    a = np.zeros(T)
    s = np.zeros(T)
    t_x = np.zeros(T)
    t_y = np.zeros(T)

    # linspace
    for i in range(num_node - 1):
        a[node[i]:node[i + 1]] = np.linspace(
            A[i], A[i + 1], node[i + 1] - node[i]) * np.pi / 180
        s[node[i]:node[i + 1]] = np.linspace(S[i], S[i + 1],
                                             node[i + 1] - node[i])
        t_x[node[i]:node[i + 1]] = np.linspace(T_x[i], T_x[i + 1],
                                               node[i + 1] - node[i])
        t_y[node[i]:node[i + 1]] = np.linspace(T_y[i], T_y[i + 1],
                                               node[i + 1] - node[i])

    theta = np.array([[np.cos(a) * s, -np.sin(a) * s],
                      [np.sin(a) * s, np.cos(a) * s]])

    # perform transformation
    for i_frame in range(T):
        xy = data_numpy[0:2, i_frame, :, :]
        new_xy = np.dot(theta[:, :, i_frame], xy.reshape(2, -1))
        new_xy[0] += t_x[i_frame]
        new_xy[1] += t_y[i_frame]
        data_numpy[0:2, i_frame, :, :] = new_xy.reshape(2, V, M)

    return data_numpy


def rand_pick(seq, probabilities):
    x = random.uniform(0, 1)
    cumprob = 0.0
    for item, item_pro in zip(seq, probabilities):
        cumprob += item_pro
        if x < cumprob:
            break
    return item


value_list = [0, 1]
probabilities = [0.5, 0.5]
keypoint_num = 16


def horizontal_flip(data_numpy):
    if rand_pick(value_list, probabilities) == 0:
        data = data_numpy[0, :, :, :]
        x_max = data.max()
        for t in range(data_numpy.shape[1]):
            for v in range(data_numpy.shape[2]):
                x_flip = 2 * x_max - data_numpy[0, t, v, :]
                data_numpy[0, t, v, :] = x_flip - x_max
        # data_numpy = torch.from_numpy(data_numpy)
        return data_numpy
    else:
        return data_numpy


def visualize(img, joints, score=None):

    # pairs = [[16, 14], [14, 12], [17, 15], [15, 13],
    #          [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
    #          [1, 2], [1, 3], [2, 4], [3, 5], [1, 19],
    #          [6, 19], [7, 19], [12, 20], [13, 20], [19, 18], [18, 20]]
    # pairs = [[16, 14], [14, 12], [17, 15], [15, 13],
    #          [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
    #          [1, 2], [1, 3], [2, 4], [3, 5], [1, 21], [10, 19],
    #          [11, 19], [18, 19], [6, 21], [7, 21], [12, 22], [13, 22], [21, 20], [20, 22]]
    # 鼻子1，左眼2，右眼3，左耳4，右耳5，左肩6，右肩7，左肘8，右肘9，左腕10，右腕11，
    # 左髋12，右髋13，左膝14，右膝15，左踝16，右踝17，杆头18，杆尾19,重心20,肩部中点21，髋部中点22
    pairs = [[12, 10], [10, 8], [13, 11], [11, 9],
             [2, 4], [3, 5], [4, 6], [5, 7], [1, 15],
             [2, 15], [3, 15], [8, 16], [9, 16], [15, 14], [14, 16]]
    # 鼻子1，左肩2，右肩3，左肘4，右肘5，左腕6，右腕7，
    # 左髋8，右髋9，左膝10，右膝11，左踝12，右踝13，重心14,肩部中点15，髋部中点16

    color = np.random.randint(0, 256, (keypoint_num, 3)).tolist()
    joints_array = np.ones((keypoint_num, 2), dtype=np.float32)
    # 将x一堆，y一堆，z一堆调整为x1,y1,z1
    joints_adjust = []
    for i in range(keypoint_num):
        joints_adjust.append(joints[i])
        joints_adjust.append(joints[i + keypoint_num])
        joints_adjust.append(joints[i + keypoint_num * 2])
    joints = joints_adjust
    for i in range(keypoint_num):
        joints_array[i, 0] = joints[i * 3]
        joints_array[i, 1] = joints[i * 3 + 1]

    for i in range(keypoint_num):
        if joints_array[i, 0] > 0 and joints_array[i, 1] > 0:
            cv2.circle(img, tuple(
                joints_array[i, :2]), 5, tuple(color[i]), 2)

    def draw_line(img, p1, p2):
        c = (0, 0, 255)
        if p1[0] > 0 and p1[1] > 0 and p2[0] > 0 and p2[1] > 0:
            cv2.line(img, tuple(p1), tuple(p2), c, 2)

    for pair in pairs:
        draw_line(img, joints_array[pair[0] - 1],
                  joints_array[pair[1] - 1])

    return img


if __name__ == "__main__":
    file_path = "/home/zqr/data/golfdb_keypoints/human_keypoints/simplify/764.json"
    view_path = "/home/zqr/data/test/dataloader_test/test_aug/764"
    ori_view_path = "/home/zqr/data/test/dataloader_test/test_aug/764_ori"
    keypoints_list = []
    with open(file_path, "r") as f:
        data_list = json.load(f)
    data_list.sort(key=lambda res: res["image_id"])
    x_data = []
    y_data = []
    z_data = []
    for elem in data_list:
        keypoints_list.append(elem["keypoints"])
    video_len = len(keypoints_list)
    for i in range(video_len):
        for j in range(16):
            x_data.append(keypoints_list[i][j * 3])
            y_data.append(keypoints_list[i][j * 3 + 1])
            z_data.append(keypoints_list[i][j * 3 + 2])
    data_adj = [x_data, y_data, z_data]
    data_numpy = np.array(data_adj)
    data_numpy = data_numpy.reshape(3, video_len, 16, 1)
    data = horizontal_flip(data_numpy)
    data = torch.from_numpy(data)

    img_index = 0
    for i in range(data.shape[1]):
        data_input = data[:, img_index, :, :]
        img = np.zeros((720, 1280, 3))
        data_input = torch.reshape(
            data_input, (keypoint_num*3,)).numpy().tolist()
        img = visualize(img, data_input)
        img_path = os.path.join(
            view_path, "{}.jpg".format(str(img_index)))
        img_index += 1
        cv2.imwrite(img_path, img)
    # img_index = 0
    # data_numpy = torch.from_numpy(data_numpy)
    # for i in range(data_numpy.shape[1]):
    #     data_input = data_numpy[:, img_index, :, :]
    #     img = np.zeros((720, 1280, 3))
    #     data_input = torch.reshape(
    #         data_input, (keypoint_num*3,)).numpy().tolist()
    #     img = visualize(img, data_input)
    #     img_path = os.path.join(
    #         ori_view_path, "{}.jpg".format(str(img_index)))
    #     img_index += 1
    #     cv2.imwrite(img_path, img)

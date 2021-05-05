import pickle
import numpy as np
import torch
import cv2
import os
import time

def get_flow(prev, cur, norm=True):
    h, w = prev.shape[0:2]
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray,
                                       None,
                                       0.6, 5, 15, 3, 5, 1.2, 0)
    flow[..., 0] = flow[..., 0] + np.arange(w)[np.newaxis, :]
    flow[..., 1] = flow[..., 1] + np.arange(h)[:, np.newaxis]
    if norm:
        flow[..., 0] = (flow[..., 0] - w/2) / (w/2)
        flow[..., 1] = (flow[..., 1] - h/2) / (h/2)
    return flow

time1 = time.time()

data_dir = '../data/kitti_mod/'

SIZE_H, SIZE_W = 128, 416

with open(data_dir + 'val.txt', 'r') as fh:
    folder_list = [i.strip() for i in fh]

M = len(folder_list)

for m in range(M):
    folder = folder_list[m]
    save_dir = '../data/flow_kitti_mod/' + folder + '/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    images = [img for img in os.listdir(data_dir + folder + '/') if img.endswith(".jpg")]
    images.sort()
    N = len(images)

    img1 = cv2.resize(cv2.imread(data_dir + folder + '/' + images[0]), (SIZE_W, SIZE_H), interpolation=cv2.INTER_AREA)

    for i in range(1, N):
        img2 = cv2.resize(cv2.imread(data_dir + folder + '/' + images[i]), (SIZE_W, SIZE_H), interpolation=cv2.INTER_AREA)

        np.save(save_dir + images[i-1][:-4] + '.npy', get_flow(img1, img2))
        np.save(save_dir + images[i][:-4] + '_r.npy', get_flow(img2, img1))

        img1 = img2

        if i % 10 == 0 and m > 0:
            print('{}/{}: {}/{}     ETA: {:.2f}min'.format(m, M, i, N, (time.time() - time1) / 60 / m * M), end='\r')



   



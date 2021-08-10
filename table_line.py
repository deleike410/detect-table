#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

from config import tableModeLinePath
from utils import letterbox_image, get_table_line, adjust_lines, line_to_line
import numpy as np
import cv2

from tabel_net import model

model.load_weights("models/custom/table-line-fine-2.h5")


def table_line(img, size=(512, 512), hprob=0.5, vprob=0.5, row=50, col=30, alph=15):
    sizew, sizeh = size
    inputBlob, fx, fy = letterbox_image(img[..., ::-1], (sizew, sizeh))
    pred = model.predict(np.array([np.array(inputBlob) / 255.0]))
    pred = pred[0]

    tmp1 = pred[..., 1]
    print(tmp1.shape)
    import matplotlib.pyplot as plt
    plt.imshow(tmp1)
    plt.show()
    tmp2 = pred[..., 0]
    plt.imshow(tmp2)
    plt.show()

    vpred = pred[..., 1] > vprob  # 竖线
    hpred = pred[..., 0] > hprob  # 横线
    vpred = vpred.astype(int)
    hpred = hpred.astype(int)

    plt.imshow(vpred)
    plt.show()
    plt.imshow(hpred)
    plt.show()


    colboxes = get_table_line(vpred, axis=1, lineW=col)
    rowboxes = get_table_line(hpred, axis=0, lineW=row)
    ccolbox = []
    crowlbox = []
    if len(rowboxes) > 0:
        rowboxes = np.array(rowboxes)
        rowboxes[:, [0, 2]] = rowboxes[:, [0, 2]] / fx
        rowboxes[:, [1, 3]] = rowboxes[:, [1, 3]] / fy
        xmin = rowboxes[:, [0, 2]].min()
        xmax = rowboxes[:, [0, 2]].max()
        ymin = rowboxes[:, [1, 3]].min()
        ymax = rowboxes[:, [1, 3]].max()
        ccolbox = [[xmin, ymin, xmin, ymax], [xmax, ymin, xmax, ymax]]
        rowboxes = rowboxes.tolist()

    if len(colboxes) > 0:
        colboxes = np.array(colboxes)
        colboxes[:, [0, 2]] = colboxes[:, [0, 2]] / fx
        colboxes[:, [1, 3]] = colboxes[:, [1, 3]] / fy

        xmin = colboxes[:, [0, 2]].min()
        xmax = colboxes[:, [0, 2]].max()
        ymin = colboxes[:, [1, 3]].min()
        ymax = colboxes[:, [1, 3]].max()
        colboxes = colboxes.tolist()
        crowlbox = [[xmin, ymin, xmax, ymin], [xmin, ymax, xmax, ymax]]

    # rowboxes += crowlbox
    # colboxes += ccolbox

    rboxes_row_, rboxes_col_ = adjust_lines(rowboxes, colboxes, alph=alph)
    rowboxes += rboxes_row_
    colboxes += rboxes_col_
    nrow = len(rowboxes)
    ncol = len(colboxes)
    for i in range(nrow):
        for j in range(ncol):
            rowboxes[i] = line_to_line(rowboxes[i], colboxes[j], 10)
            colboxes[j] = line_to_line(colboxes[j], rowboxes[i], 10)

    return rowboxes, colboxes


if __name__ == '__main__':
    import time

    p = 'img/成绩.png'
    from utils import draw_lines

    img = cv2.imread(p)
    t = time.time()

    rowboxes, colboxes = table_line(img[..., ::-1], size=(512, 512), hprob=0.5, vprob=0.5)
    img = draw_lines(img, rowboxes + colboxes, color=(255, 0, 0), lineW=1)

    print(time.time() - t, len(rowboxes), len(colboxes))
    cv2.imwrite('img/成绩-line.png', img)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import time
import cv2
from functools import reduce
from operator import mul

from config import tableModeLinePath
from utils import letterbox_image, get_table_line, adjust_lines, line_to_line
from tabel_net import model
from utils import draw_lines


def table_line(img, size=(512, 512), hprob=0.5, vprob=0.5, row=50, col=30, alph=15):
    sizew, sizeh = size
    inputBlob, fx, fy = letterbox_image(img[..., ::-1], (sizew, sizeh))
    model.load_weights("models/custom/table-line-fine-2.h5")
    pred = model.predict(np.array([np.array(inputBlob) / 255.0]))
    pred = pred[0]

    # tmp1 = pred[..., 1]
    # print(tmp1.shape)
    # import matplotlib.pyplot as plt
    # plt.imshow(tmp1)
    # plt.show()
    # tmp2 = pred[..., 0]
    # plt.imshow(tmp2)
    # plt.show()

    vpred = pred[..., 1] > vprob
    hpred = pred[..., 0] > hprob
    vpred = vpred.astype(int)
    hpred = hpred.astype(int)

    # plt.imshow(vpred)
    # plt.show()
    # plt.imshow(hpred)
    # plt.show()

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

    # rowboxes += crowlbox   # remove later
    # colboxes += ccolbox

    rboxes_row_, rboxes_col_ = adjust_lines(rowboxes, colboxes, alph=alph)
    rowboxes += rboxes_row_
    colboxes += rboxes_col_
    nrow = len(rowboxes)
    ncol = len(colboxes)

    intersection_points = []

    for i in range(nrow):
        row_intersection = []
        for j in range(ncol):
            row_p = line_to_line(rowboxes[i], colboxes[j], 10)
            rowboxes[i] = row_p[0]
            row_intersection.append([row_p[1][0], row_p[1][1]])

            col_p = line_to_line(colboxes[j], rowboxes[i], 10)
            colboxes[j] = col_p[0]
            # intersection_points.append(col_p[1])
        intersection_points.append(row_intersection)

    return rowboxes, colboxes, intersection_points


def reshape(lst, shape):
    if len(shape) == 1:
        return lst
    n = reduce(mul, shape[1:])
    return [reshape(lst[i * n:(i + 1) * n], shape[1:]) for i in range(len(lst) // n)]


def get_rect(intersection_points, y_max, x_max, rows, cols):
    # sort
    res = [sorted(i, key=lambda x: x[0]) for i in intersection_points]
    rects = [[0] * (cols - 1) for _ in range(rows - 1)]
    for i in range(len(rects)):
        for j in range(len(rects[0])):
            tmp = [res[i][j], res[i][j + 1], res[i + 1][j], res[i + 1][j + 1]]
            rects[i][j] = tmp

    return rects


if __name__ == '__main__':
    p = 'img/成绩.png'
    img = cv2.imread(p)
    t = time.time()

    rowboxes, colboxes, intersection_points = table_line(img[..., ::-1], size=(512, 512), hprob=0.5, vprob=0.5)
    img2 = img
    image_h, image_w = img.shape[:2]
    rects = get_rect(intersection_points, image_h, image_w, len(rowboxes), len(colboxes))

    for rec in rects:
        for p in rec:
            cv2.rectangle(img2, (int(p[0][0]), int(p[0][1])), (int(p[3][0]), int(p[3][1])), (0, 255, 0), 1)
            cv2.circle(img2, (int(p[0][0]), int(p[0][1])), radius=2, color=(0, 0, 255), thickness=2)
            cv2.circle(img2, (int(p[3][0]), int(p[3][1])), radius=2, color=(0, 0, 255), thickness=2)

    cv2.imwrite('img/成绩_rect.jpg', img2, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    img = draw_lines(img, rowboxes + colboxes, color=(255, 0, 0), lineW=1)
    print(time.time() - t, len(rowboxes), len(colboxes))
    cv2.imwrite('img/成绩_line.png', img)

#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import cv2 as cv
import json

import base64
from io import BytesIO
from PIL import Image, ImageFont, ImageDraw


def PIL_to_base64(image):
    output = BytesIO()
    image.save(output, format='png')
    contents = output.getvalue()
    output.close()
    string = str(base64.b64encode(contents))
    return string


def paint_chinese_opencv(im, chinese, pos, color):
    fillColor = color  # (255,0,0)
    position = pos  # (100,100)

    img_PIL = Image.fromarray(cv.cvtColor(im, cv.COLOR_BGR2RGB))
    font = ImageFont.truetype('/usr/share/fonts/opentype/noto/simsun.ttf', 15, layout_engine=ImageFont.LAYOUT_BASIC)

    draw = ImageDraw.Draw(img_PIL)
    draw.text(position, chinese, font=font, fill=fillColor)

    img = cv.cvtColor(np.asarray(img_PIL),cv.COLOR_RGB2BGR)
    return img


def generate_img_json(height, width, index, words):
    # height = 1024
    # width = 720
    img = np.zeros((height, width, 3), np.uint8) + 255  # 生成一个白色图像

    # 起点和终点的坐标
    ends = np.arange(height)[::30][1:].tolist()
    starts = np.arange(width)[::70].tolist()
    starts[0] = ends[0]
    padding = 50

    lines = []
    labels = []
    anno = []
    # 横线
    for end in ends:
        ptStart = (30, end)
        ptEnd = (starts[-1], end)
        # ptend[0] 0.15的概率随机
        if np.random.randint(0, 100) > 85:
            early_end = int(np.random.uniform(int(starts[-1] * 0.3), starts[-1]))
        else:
            early_end = starts[-1]
        ptEnd = (early_end, end)
        tmp = [list(ptStart), list(ptEnd)]
        lines.append(tmp)
        labels.append('0')
        anno.append({
            'label': "0",
            'line_color': [0, 0, 128],
            'fill_color': [0, 0, 128],
            'points': tmp,
            'shape_type': "line",
            'flags': {}
        })
        point_color = (0, 0, 128)  # BGR
        thickness = 1
        lineType = 4
        cv.line(img, ptStart, ptEnd, point_color, thickness, lineType)
        text_start = ptStart[0]
        while text_start < ptEnd[0]:
            # 绘制汉字数字
            if np.random.randint(0, 100) > 60:
                random_index = [np.random.randint(0, len(words)) for _ in range(3)]
                text = "".join([words[i] for i in random_index])
            else:
                text = "".join([str(np.random.randint(0, 10)) for _ in range(4)])
            img = paint_chinese_opencv(img, chinese=text, pos=(text_start, ptStart[1] + 5), color=(0, 0, 0))
            text_start += np.random.randint(50, 55)

    for start in starts:
        if start > ends[-1]:
            continue
        ptStart = (start, 30)
        ptEnd = (start, ends[-1])
        # ptend[0] 0.15的概率随机
        if np.random.randint(0, 100) > 85:
            early_end = int(np.random.uniform(int(ends[-1] * 0.3), ends[-1]))
        else:
            early_end = ends[-1]
        ptEnd = (start, early_end)
        tmp = [list(ptStart), list(ptEnd)]
        lines.append(tmp)
        labels.append('1')
        anno.append({
            'label': "1",
            'line_color': [0, 0, 128],
            'fill_color': [0, 0, 128],
            'points': tmp,
            'shape_type': "line",
            'flags': {}
        })
        point_color = (0, 0, 128)  # BGR
        thickness = 1
        lineType = 4
        cv.line(img, ptStart, ptEnd, point_color, thickness, lineType)


    save_path = 'images/' + str(index).zfill(8) + '.png'
    cv.imwrite(save_path, img)

    # 生成coco类型 json标注
    pil_img = Image.open(save_path)
    data = {'version': 'sq1.0',
            'flags': {},
            'shapes': anno,
            'imageData': PIL_to_base64(image=pil_img),
            'lineColor': [0, 255, 0, 128],
            'fillColor': ([255, 0, 0, 128],),
            'imagePath': save_path,
            }
    jsonString = json.dumps(data)
    save_json = 'images/' + str(index).zfill(8) + '.json'
    f2 = open(save_json, 'w')
    f2.write(jsonString)
    f2.close()


if __name__ == '__main__':
    with open('hanzi.txt', 'r') as f:
        words = f.readline()
    heights = np.arange(640, 1024)[::5].tolist()
    widths = np.arange(360, 720)[::5].tolist()
    count = 0
    # 5008 / 2 = 2504
    for height in heights:
        for width in widths:
            generate_img_json(height, width, count, words)
            count += 1
    print(count)

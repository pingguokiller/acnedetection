import os
import json
import numpy as np
import glob
import shutil
from sklearn.model_selection import train_test_split
import tqdm

np.random.seed(41)

import math
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import copy

import labelme.utils as lu


# 从labelme文件中的base64数据中获取图像数据
def originImageLoader(image_path):
    with open(image_path, encoding='utf-8') as json_file:
        labelmeJson = json.load(json_file)

        image = lu.img_b64_to_arr(labelmeJson['imageData'])
        image = image.astype(np.uint8)

        return image


# 0为背景
dataset = 'skin'  # face, skin

if dataset == 'face':
    classname_to_id = {"brow": 1, "nose": 2, "mouth": 3}
    labelme_path = "E:/dataset/face/labelme/"
    saved_coco_path = "E:/dataset/face/"
elif dataset == 'skin': # COCO里面的category_id范围是 1~10
    classname_to_id = {"closed_comedo": 1, "open_comedo": 2, "papule": 3, "pustule": 4, "nodule": 5,
                       "atrophic_scar": 6, "hypertrophic_scar": 7, "melasma": 8, "nevus": 9, "other": 10}
    labelme_path = "/home/zhangjw/research/skin_data/labelme"
    saved_coco_path = "/home/zhangjw/research/skin_data/"


# 读取json文件，返回一个json对象
def read_jsonfile(path):
    with open(path, "r", encoding='utf-8') as f:
        return json.load(f)

# 主程序入口
# 创建存放标注文件的目录
# 获取labemel目录下所有的json文件列表
json_path_list = glob.glob(os.path.join(labelme_path, "*.json"))
print('json_path_list:', len(json_path_list))

# 把训练集转化为COCO的json格式
print('begin to check!')
# 遍历所有标注数据
for json_path in tqdm.tqdm(json_path_list):
    # labelme_json 标注数据
    labelme_json = read_jsonfile(json_path)

    # 遍历labelme_json中所有的形状标注
    for shape in labelme_json['shapes']:
        # 将圆形标注转化为多边形标注，然后统一处理
        if shape['shape_type'] == "circle":
            points = shape['points']
            xy = [tuple(point) for point in points]
            (cx, cy), (px, py) = xy  # 圆心和第二点
            # print(xy)
            # TODO：将圆形转换为多边形
            d = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
            d1 = math.sqrt((d ** 2) / 2)
            points = [[cx, cy - d], [cx - d1, cy - d1], [cx - d, cy], [cx - d1, cy + d1],
                      [cx, cy + d], [cx + d1, cy + d1], [cx + d, cy], [cx + d1, cy - d1]]
            #                 print(points)
            shape['points'] = points


        # 判断点的个数，如果不超过2个，那么该标注信息错误
        # 点的个数 [(coor_x, coor_y), ...]
        effective_cnt = len(shape['points'])
        # 如果在范围内地点小于等于2个
        if effective_cnt <= 2:
            print('json_path:%s has wrong labelinfos!' % (json_path), )
            print(shape['points'])
            print('shape:', shape)
            continue
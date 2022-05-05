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


# 主程序入口
# 创建存放标注文件的目录
coco_annotation_dir = os.path.join(saved_coco_path, "coco/annotations/")

# train标注数据文件路径
coco_train_annotation_path = os.path.join(saved_coco_path, 'coco/annotations/instances_train2017.json')
# val标注数据文件数据
coco_val_annotation_path = os.path.join(saved_coco_path, 'coco/annotations/instances_val2017.json')



# 获取labemel目录下所有的json文件列表
json_path_list = glob.glob(os.path.join(labelme_path, "*.json"))
# 数据划分,这里没有区分val2017和tran2017目录，所有图片都放在images目录下
train_path_list, val_path_list = train_test_split(json_path_list, test_size=0.1)

print('val_path_list:', len(val_path_list))
print('val_path_list:')
for x in val_path_list:
    print(x)




# 获取labemel目录下所有的json文件列表
img_path_list = glob.glob(os.path.join(saved_coco_path, 'coco/images/test2017_whole', "*.jpg"))
# 数据划分,这里没有区分val2017和tran2017目录，所有图片都放在images目录下
train_img_path_list, val_img_path_list = train_test_split(img_path_list, test_size=0.1)

print('val_img_path_list:', len(val_img_path_list))
print('val_img_path_list:')
for x in val_img_path_list:
    print(x)
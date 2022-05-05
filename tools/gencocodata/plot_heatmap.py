import pandas as pd
import pymysql
import PIL.Image
import base64
import json
import io
import os
import shutil
from tqdm import tqdm
import numpy as np
import glob
import matplotlib.pyplot as plt


# 读取json文件，返回一个json对象
def read_jsonfile(path):
    with open(path, "r", encoding='utf-8') as f:
        return json.load(f)


# 构建COCO的image字段
def _image(obj, path):
    '''
    构建COCO的image字段：
    读取宽、高、名称，赋给id
    '''
    image = {}
    # from labelme import utils
    # img_x = utils.img_b64_to_arr(obj['imageData'])

    # h, w = img_x.shape[:-1]
    image['height'] = obj['imageHeight']
    image['width'] = obj['imageWidth']
    image['file_name'] = os.path.basename(path).replace(".json", ".jpg")
    return image


# 获取labemel目录下所有的json文件列表
all_path_list = glob.glob(os.path.join('/home/zhangjw/research/skin_data/labelme', "*.json"))

label_2_means = {}
for json_index, json_path in enumerate(all_path_list[:]):
    # 获取annotation信息
    obj = read_jsonfile(json_path)

    label_list = obj['shapes']
    img_height = obj['imageHeight']
    img_width = obj['imageWidth']

    # 保存图像
    # img_numpy = originImageLoader(json_path)
    # im = Image.fromarray(img_numpy)
    # im.save(, quality = 95)

    # print('shapes:', shapes)
    # print('img_height, img_width:', img_height, img_width)

    # 遍历一个图像的标注
    for label_info in label_list:
        label_type = label_info['label']
        plist = label_info['points']

        p_array = np.array(plist)
        # print('p_array:', p_array.shape)
        cor_mean = np.mean(p_array, axis=0)

        if label_type in label_2_means.keys():
            label_2_means[label_type].append([cor_mean[0] / img_width, cor_mean[1] / img_height])
        else:
            label_2_means[label_type] = []
            label_2_means[label_type].append([cor_mean[0] / img_width, cor_mean[1] / img_height])

#  画图
cor_mean_list = []
for label_type in label_2_means.keys():
    plists = label_2_means[label_type]

    # 针对每个标注
    for cor_mean in plists:
        cor_mean_list.append(cor_mean)

cor_mean_array = np.array(cor_mean_list)
print('cor_mean_array:', cor_mean_array.shape)





plt.figure(figsize=(8, 10))
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.scatter(cor_mean_array[:, 0], 1 - cor_mean_array[:, 1])
plt.show()

grid_size = 100
count_array = np.zeros((grid_size, grid_size))

for i in range(cor_mean_array.shape[0]):
    x = cor_mean_array[i, 0]
    y = cor_mean_array[i, 1]

    x_index = int(x // (1 / grid_size))
    y_index = int((1 - y) // (1 / grid_size))

    count_array[y_index, x_index] += 1

plt.figure(figsize=(8, 10))
plt.imshow(count_array, interpolation='nearest', cmap=plt.cm.jet, origin='lower')
plt.colorbar(shrink=.92)
plt.show()

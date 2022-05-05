#!/usr/bin/env python

import pandas as pd
import pymysql
import PIL.Image
import base64
import json
import sys
import io
import os
import shutil
from tqdm import tqdm

from img_name_list import origin_img_name_list


# get data from mysql database of sever
def get_data():
    # 连接数据库信息
    database = pymysql.connect(
        host="192.168.7.197",
        database='mysql',
        port=4322,
        user='root',
        password='labelskinmachineilab',
    )

    # TODO: 是否是1号任务？是否查询全部终审完毕的图片？终审员是否为7号或有多个终审员？
    # 查询所需的数据
    sql = 'SELECT imginfo.imgfile, assigninfo.labelinfo \
        FROM skin.labelapi_imginfo as imginfo, skin.labelapi_matchinfo as matchinfo, skin.labelapi_assigninfo as assigninfo \
        where imginfo.id = matchinfo.imginfo_id and matchinfo.id = assigninfo.matchinfo_id \
        and matchinfo.taskinfo_id = 1 and matchinfo.status = 3 and assigninfo.userinfo_id = 7'

    # 处理所需的数据
    data_df = pd.read_sql(sql, database)
    imgfile_list = data_df['imgfile'].tolist()
    labelinfo_list = data_df['labelinfo'].tolist()
    return imgfile_list, labelinfo_list


def build_image_b64(img_path):
    img_pil = PIL.Image.open(img_path)
    width, height = img_pil.size
    # f = io.BytesIO()
    # img_pil.save(f, format="JPEG")
    # img_bin = f.getvalue()
    # img_b64 = base64.b64encode(img_bin)
    # img_b64 = str(img_b64, encoding='utf-8')
    return width, height #img_b64,


def build_label_list(label_str):
    class_name_to_id = {"closed_comedo": 1, "open_comedo": 2, "papule": 3, "pustule": 4, "nodule": 5,
        "atrophic_scar": 6, "hypertrophic_scar": 7, "melasma": 8, "nevus": 9, "other": 10}

    label_str = label_str.replace('\'', '\"')
    label_str = label_str.replace('None', 'null')
    #print('label_str:', label_str[32760:32770])
    label_list = json.loads(label_str)

    for instance in label_list:
        if 'uuid' in instance:
            instance.pop('uuid')
        for key, value in class_name_to_id.items():
            if instance['label'] == value:
                instance['label'] = key
                break
    return label_list


# main
base_dir = '/home/zhangjw/research/skin_data' # 服务器数据集根目录

image_dir = os.path.join(base_dir, 'raw')
labelme_dir = os.path.join(base_dir, 'labelme')

#
if os.path.exists(labelme_dir):
    shutil.rmtree(labelme_dir)

os.makedirs(labelme_dir)

# get the img list and label_str list from database
image_file_list, label_str_list = get_data()
sample_num = len(image_file_list)
print('building labelme info sample_num:', sample_num)

#
for i in tqdm(range(sample_num)):
    image_file = image_file_list[i].split('/')[-1]
    label_str = label_str_list[i]

    # label info
    # if image_file!='JX__何宇航_痤疮_20200623083830000_斑点.jpg':
    #     continue
    # else:
    #     print('label_str:', label_str[32760:32770])

    # 只保留之前276张图片的优质标签
    if image_file.split('.')[0] not in origin_img_name_list:
        continue

    # raw image path
    image_path = os.path.join(image_dir, image_file)
    # img info
    width, height = build_image_b64(image_path) # image_b64,
    print('image_file:', image_file)
    label_list = build_label_list(label_str)

    #
    labelme_dict = {
        "version": "4.5.6",
        "flags": {},
        "shapes": label_list,
        "imagePath": "",
        #"imageData": image_b64,
        "imageHeight": height,
        "imageWidth": width,
    }

    # TODO: 覆盖原文件还是追加或者新增等等？？？
    labelme_file = image_file.split('.')[0] + '.json'
    labelme_path = os.path.join(labelme_dir, labelme_file)
    with open(labelme_path, 'w', encoding='utf-8') as f:
        json.dump(labelme_dict, f, indent=2)



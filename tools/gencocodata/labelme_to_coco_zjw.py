import os
import json
import numpy as np
np.random.seed(41)
import glob
import shutil
from sklearn.model_selection import train_test_split
import tqdm
import math
import matplotlib.pyplot as plt
import cv2
import copy
from PIL import Image
import argparse


# 检验是否全是中文字符
def is_all_chinese(strs):
    if len(strs) <= 0:
        return False

    for _char in strs:
        if not '\u4e00' <= _char <= '\u9fa5':
            return False

    return True

# 统计 json名称, 用于约束验证集图像选择，只选择拍摄了一次照片的人
def sta_path_list(val_path_list):
    # print('len(val_path_list):', len(val_path_list))

    all_ID_list = []
    all_ID_dict = {}

    all_name_list = []
    all_name_dict = {}

    #
    for path_index, train_path in enumerate(val_path_list):
        tmp_part_list = os.path.basename(train_path).split('_')
        for part_index, tmp_part in enumerate(tmp_part_list):
            if tmp_part != '' and tmp_part[0] == '2':
                # print(tmp_part)
                all_ID_list.append(tmp_part)
                if tmp_part not in all_ID_dict.keys():
                    all_ID_dict[tmp_part] = 1
                else:
                    all_ID_dict[tmp_part] += 1

            if is_all_chinese(tmp_part) and tmp_part[0] not in ['痤', '斑', '标', '玫']:
                if len(tmp_part) > 4:
                    tmp_name = tmp_part[:-2]
                else:
                    tmp_name = tmp_part
                # print(tmp_name)
                all_name_list.append(tmp_name)

                if tmp_name not in all_name_dict.keys():
                    all_name_dict[tmp_name] = {'name': tmp_name, 'num': 1, 'order_index': path_index}
                else:
                    all_name_dict[tmp_name] = {'name': tmp_name, 'num': 1 + all_name_dict[tmp_name]['num'],
                                               'order_index': None}

    # print('len(all_ID_list):', len(all_ID_list))
    # print('len(all_ID_dict):', len(all_ID_dict))
    # print('len(all_name_list):', len(all_name_list))
    # print('len(all_name_dict):', len(all_name_dict))
    return all_ID_list, all_ID_dict, all_name_list, all_name_dict

# 根据人的名称划分数据集
def split_train_val_by_name(json_path_list):
    all_ID_list, all_ID_dict, all_name_list, all_name_dict = sta_path_list(json_path_list)
    if len(all_ID_list) != len(all_name_list):
        exit('判断是否为名字错误')

    # print(len(all_ID_list), len(all_name_list)) # , all_name_dict
    # 所有的图片数量
    # print('all_pic_num:', len(json_path_list))

    # 所有的姓名数量
    all_name_num = len(all_name_list)
    # print('all_name_num:', all_name_num)

    # 验证集需要多少个图片
    val_num = np.round(0.1 * all_name_num).astype(np.int8)
    # print('val_num:', val_num)

    # 只有一个图片的姓名列表
    all_name_1_list = []
    all_name_1_jsonindex_list = []
    for name in all_name_dict.keys():
        info = all_name_dict[name]
        if info['num'] == 1:
            all_name_1_list.append(info)
            all_name_1_jsonindex_list.append(info['order_index'])
        # print(name, num)

    # print('name_1_num:', len(all_name_1_list))

    # 从 all_name_1_list 中随机找出 val_num 个

    # 将这些姓名以0.85:0.15 train:test
    train_jonsindex_list, val_jonsindex_list = train_test_split(all_name_1_jsonindex_list,
                                                                test_size=val_num / len(all_name_1_jsonindex_list))
    # print('val_jonsindex_list:', val_jonsindex_list)

    # 最终确定训练json 验证json
    train_jsonpath_list = []
    val_jsonpath_list = []
    for json_index, json_path in enumerate(json_path_list):
        if json_index in val_jonsindex_list:
            val_jsonpath_list.append(json_path)
        else:
            train_jsonpath_list.append(json_path)

    print('len(train_jsonpath_list):', len(train_jsonpath_list))
    print('len(val_jsonpath_list):', len(val_jsonpath_list))

    return train_jsonpath_list, val_jsonpath_list



# 点类
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


# 计算是否overlap， Returns true if two rectangles(l1, r1) and (l2, r2) overlap
def doOverlap(l1, r1, l2, r2):
    # To check if either rectangle is actually a line
    # For example  :  l1 ={-1,0}  r1={1,1}  l2={0,-1}  r2={0,1}

    if (l1.x == r1.x or l1.y == r1.y or l2.x == r2.x or l2.y == r2.y):
        # the line cannot have positive overlap
        return False

    # If one rectangle is on left side of other
    if (l1.x >= r2.x or l2.x >= r1.x):
        return False

    # If one rectangle is above other
    if (r1.y <= l2.y or r2.y <= l1.y):
        return False

    return True



# 标注数据格式转换类
class Lableme2CoCo:
    def __init__(self, saved_coco_path="/home/zhangjw/research/skin_data/"):
        # coco 数据集保存根目录
        self.saved_coco_path = saved_coco_path

        # # COCO里面的category_id范围是 1~10
        self.classname_to_id = {"closed_comedo": 1, "open_comedo": 2, "papule": 3, "pustule": 4, "nodule": 5,
            "atrophic_scar": 6, "hypertrophic_scar": 7, "melasma": 8, "nevus": 9, "other": 10}

        #
        self.images = []
        self.annotations = []
        self.categories = []
        self.img_id = 0
        self.ann_id = 0

        # 生成 cat_id, cat_name 数组
        self._init_categories()

    # 根据多边形点坐标置黑， return masked_img
    def draw_mask_by_polycorlist(self, img, cor_list, mask_flag=True):
        if mask_flag:
            pts = np.array(cor_list, np.int32)  # 数据类型必须为 int32
            pts = pts.reshape((-1, 1, 2))

            cv2.polylines(img, np.int32([pts]), 1, 0)
            cv2.fillPoly(img, np.int32([pts]), 0)

        return img

    # 从labelme文件中的base64数据中获取图像数据; 直接从原始文件raw中获取文件
    def originImageLoader(self, json_path):
        raw_dir = os.path.join(self.saved_coco_path, 'raw')
        image_path = os.path.join(raw_dir, os.path.basename(json_path).replace(".json", ".jpg"))

        if os.path.exists(image_path):
            img_pil = Image.open(image_path)
            image = np.array(img_pil).astype(np.uint8)
            return image
        else:
            print("image_path: %s not exist!" % (image_path))
            exit()

    #
    def save_coco_json(self, instance, save_path):
        json.dump(instance, open(save_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)  # indent=2 更加美观显示

    # 将labelme 数据集转化为 coco数据集
    def to_coco(self, json_path_list, img_save_pre):
        '''
        由json文件构建COCO，不包括数据强化augment
        '''
        for json_index, json_path in enumerate(tqdm.tqdm(json_path_list)):

            # 获取annotation信息
            obj = self.read_jsonfile(json_path)
            image_info = self._image(obj, json_path)

            # 保存图像
            # img_numpy = originImageLoader(json_path)
            # im = Image.fromarray(img_numpy)
            # im.save(, quality = 95)

            raw_dir = os.path.join(self.saved_coco_path, 'raw')
            src_image_path = os.path.join(raw_dir, os.path.basename(json_path).replace(".json", ".jpg"))
            target_img_path = os.path.join(img_save_pre, image_info['file_name'])
            shutil.copyfile(src_image_path, target_img_path)

            # 处理图像信息
            self.images.append(image_info)
            # 处理标注信息
            shapes = obj['shapes']
            for shape in shapes:
                annotation = self._annotation(shape, self.img_id)
                self.annotations.append(annotation)
                self.ann_id += 1
            self.img_id += 1

        instance = {}
        instance['info'] = 'spytensor created'
        instance['license'] = ['license']
        instance['images'] = self.images
        instance['annotations'] = self.annotations
        instance['categories'] = self.categories
        return instance

    # 对数据进行crop增广，并生成coco数据
    def to_aug_coco(self, json_path_list, img_save_pre, mask_flag=True):
        '''
        由json文件构建COCO，不包括数据强化augment
        '''
        print('to_aug_coco start')

        # 遍历所有标注数据
        for json_path in tqdm.tqdm(json_path_list):
            # print('handling: ', json_path)
            # 将一张大图 分为 crop 为多个小图
            sub_image_list, sub_annotation_list = self._relative_crop(json_path=json_path, img_save_pre=img_save_pre, mask_flag=mask_flag)
            self.images += sub_image_list
            self.annotations += sub_annotation_list

        instance = {}
        instance['info'] = 'spytensor created'
        instance['license'] = ['license']
        instance['images'] = self.images
        instance['annotations'] = self.annotations
        instance['categories'] = self.categories
        return instance

    # 剪切
    def _relative_crop(self, json_path, img_save_pre, img_scale=(1024, 1024), mask_flag=True):
        '''
        相对剪裁，类似于patch-based（好像是吧）,同时保存
        json_path: labelme格式json文件path
        img_save_pre: 图像保存位置
        img_scale: crop图像尺寸
        '''

        # for test
        #json_path = '/Users/zhangjianwei/Documents/skin/skin_data/labelme/J__陈雪峰_痤疮_20190305102113000_斑点.json'

        # labelme_json 标注数据
        labelme_json = self.read_jsonfile(json_path)

        # 所有小图片list
        sub_image_list = []
        #
        sub_annotation_list = []
        # 所有左上角坐标list
        leftup_cor_list = []

        # 构建COCO的image字段
        image_info = self._image(labelme_json, json_path)

        # 处理图像信息, ceil:向上取整, 表示总共有多个块
        num_h, num_w = math.ceil(image_info['height'] / img_scale[0]), math.ceil(image_info['width'] / img_scale[1])
        # 每次crop的偏移尺寸
        space_h = (image_info['height'] - img_scale[0]) / (num_h - 1)
        space_w = (image_info['width'] - img_scale[1]) / (num_w - 1)

        #
        start_img_id = self.img_id

        # 从labelme文件中的base64数据中获取图像数据
        img_numpy = self.originImageLoader(json_path)
        # 显示标注与图像
        # img_whole = img_numpy.copy()
        # shapes = obj['shapes']
        # for shape in shapes:
        #     [x, y, w, h] = self._get_box(shape['points'])
        #     left_top = (int(x), int(y))
        #     right_bottom = (int(x+w), int(y+h))
        #     cv2.rectangle(img_whole, left_top, right_bottom, (0,255,0), thickness=3)
        # plt.figure(figsize=(16,12))
        # plt.imshow(img_whole)
        # plt.show()

        # print('image_info', image_info['height'], ',', image_info['width'], 'img_numpy: ', img_numpy.shape)

        # plt.figure(figsize=(16, 12))
        # plt.imshow(img_numpy)
        # plt.show()

        # 遍历图像, 现将大图片进行切片，分为若干个小图片存入sub_image_list，并记录下所有块的左上角坐标
        sub_img_numpy_list = []
        sub_img_name_list = []
        for block_h_index in range(num_h):
            for block_w_index in range(num_w):
                sub_image = {}
                sub_image['height'], sub_image['width'] = img_scale[0], img_scale[1]
                sub_image['id'] = self.img_id + block_h_index * num_w + block_w_index
                sub_image['file_name'] = os.path.basename(json_path).replace(".json", "") + '_crop' + str(block_h_index * num_w + block_w_index) + ".jpg"
                # 额外添加的信息，代表相对原图分割的左上角位置
                sub_image['crop_xy'] = [math.ceil(block_w_index * space_w), math.ceil(block_h_index * space_h)]

                # 保存图像, 图像的第一个纬度是行（高）
                sub_img_numpy = copy.deepcopy(img_numpy[sub_image['crop_xy'][1]:sub_image['crop_xy'][1] + img_scale[1],
                                     sub_image['crop_xy'][0]:sub_image['crop_xy'][0] + img_scale[0], :])
                sub_img_name = os.path.join(img_save_pre, sub_image['file_name'])
                sub_img_numpy_list.append(sub_img_numpy)
                sub_img_name_list.append(sub_img_name)
                #im = Image.fromarray(sub_img_numpy)
                #im.save(sub_img_name, quality=95)

                #
                sub_image_list.append(sub_image)
                # 临时存储每个块的位置，包括坐上角点位和h,w
                leftup_cor_list.append((sub_image['crop_xy'][0], sub_image['crop_xy'][1]))

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

            # 获取多边形标注的bbox信息
            [x, y, w, h] = self._get_box(shape['points'])
            #print('shape[points]:', shape['points'])
            #print('[x, y, w, h]:', [x, y, w, h])

            # 错的
            # plt.figure(figsize=(12, 9))
            # plt.imshow(img_numpy[x:x+w, y:y+h, :])
            # plt.show()
            # 对的
            # plt.figure(figsize=(12, 9))
            # plt.imshow(img_numpy[y:y+h, x:x+w, :])
            # plt.show()
            # input()

            # 判断点的个数，如果不超过2个，那么该标注信息错误
            # 点的个数 [(coor_x, coor_y), ...]
            effective_cnt = len(shape['points'])
            # 如果在范围内地点小于等于2个
            if effective_cnt <= 2:
                print('json_path:%s has wrong labelinfos!'%(json_path))
                print(shape['points'])
                continue

            # 遍历所有的小图片
            for block_index, leftup_cor in enumerate(leftup_cor_list):
                #print('leftup_cor, img_scale:', leftup_cor, img_scale)
                #print('x y w h:', x,y,w,h)
                #print(doOverlap(Point(x, y), Point(x+w, y+h), Point(leftup_cor[0], leftup_cor[1]), Point(leftup_cor[0] + img_scale[0], leftup_cor[1] + img_scale[1])))
                # 如果该标注信息在该block的范围内
                if doOverlap(Point(x, y), Point(x+w, y+h), Point(leftup_cor[0], leftup_cor[1]), Point(leftup_cor[0] + img_scale[0], leftup_cor[1] + img_scale[1])):
                    # 不能用.copy(), 这个特直接用copy根本就没有复制子对象
                    shapes_item = copy.deepcopy(shape)

                    # 将大图坐标转换为小图坐标 point: (x, y) (水平坐标，垂直坐标)
                    bbox_split_flag = False
                    for point_index, point in enumerate(shapes_item['points']):
                        point[0] -= leftup_cor[0]
                        point[1] -= leftup_cor[1]

                        # 标注的边框可能不能和图像边框重合, 这儿判断一个bbox是否被拆分
                        if point[0] < 1 or point[0] > (img_scale[0] - 1) or point[1] < 1 or point[1] > (img_scale[1] - 1):
                            bbox_split_flag = True

                        point[0] = np.clip(point[0], 1, img_scale[0] - 1) # x
                        point[1] = np.clip(point[1], 1, img_scale[1] - 1) # y

                        shapes_item['points'][point_index] = point

                    # 该处标注不要且应该将原图该部分置为黑色
                    if bbox_split_flag:
                        # 画图
                        sub_img_numpy_list[block_index] = self.draw_mask_by_polycorlist(sub_img_numpy_list[block_index], shapes_item['points'],  mask_flag=mask_flag)

                        # im = Image.fromarray(sub_img_numpy)
                        # im.save(sub_img_name, quality=95)
                        # pass
                    else:
                        # 转换为coco的annotaiton格式
                        #print('shapes_item:', shapes_item)
                        annotation = self._annotation(shapes_item, start_img_id + block_index)

                        # 对于太小的标记，不考虑, w h 的限制再提升一点儿, 放大到 15
                        if annotation['bbox'][2] < 10 or annotation['bbox'][3] < 10:
                            continue

                        #print('annotation:', annotation)
                        sub_annotation_list.append(annotation)
                        self.ann_id += 1

        # 最终画图
        for block_index in range(len(sub_img_numpy_list)):
            sub_img_numpy = sub_img_numpy_list[block_index]
            sub_img_name = sub_img_name_list[block_index]
            im = Image.fromarray(sub_img_numpy)
            im.save(sub_img_name, quality=95)

        self.img_id = start_img_id + num_h * num_w

        # 调试用
        #self._show_image(img_numpy, sub_image_list, sub_annotation_list, (num_w, num_h))

        return sub_image_list, sub_annotation_list

    # 调试时显示图片用
    def _show_image(self, img_numpy, sub_image_list, sub_annotation_list, block_num, img_scale=(1024, 1024)):
        '''
        通过coco数据格式显示标注与图像
        '''
        (num_w, num_h) = block_num
        #plt.figure(figsize=(16, 12))
        for sub_image_index, image in enumerate(sub_image_list):
            #print('sub_image_index:', sub_image_index)
            im = img_numpy[image['crop_xy'][1]:image['crop_xy'][1] + img_scale[1],
                 image['crop_xy'][0]:image['crop_xy'][0] + img_scale[0], :]

            for annotation in sub_annotation_list:
                #rint('image_id:', annotation['image_id'], image['id'])
                if annotation['image_id'] == image['id']:
                    # print(annotation['segmentation'])
                    points = np.array(annotation['segmentation']).reshape(-1, 2).tolist()
                    [x, y, w, h] = self._get_box(points)
                    left_top = (int(x), int(y))
                    right_bottom = (int(x + w), int(y + h))
                    cv2.rectangle(im, left_top, right_bottom, (0, 255, 0), thickness=3)

            #plt.subplot(num_h, num_w, k + 1), # matplotlib RGB正统， cv反骨仔
            if image['id'] == 14:
                plt.figure(figsize=(12, 9))
                plt.imshow(im)
                plt.show()
                print(im.shape)
                input()
                exit()

            #plt.xticks([])  # 去掉横坐标值
            #plt.yticks([])  # 去掉纵坐标值
        # plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=0)
        #plt.subplots_adjust(wspace=-0.6, hspace=0.1)
        #plt.show()

    # 构建类别
    def _init_categories(self):
        '''
        根据全局变量classname_to_id构建类别
        '''
        for k, v in self.classname_to_id.items():
            category = {}
            category['id'] = v
            category['name'] = k
            self.categories.append(category)

    # 构建COCO的image字段
    def _image(self, obj, path):
        '''
        构建COCO的image字段：
        读取宽、高、名称，赋给id
        '''
        image = {}
        #from labelme import utils
        #img_x = utils.img_b64_to_arr(obj['imageData'])

        #h, w = img_x.shape[:-1]
        image['height'] = obj['imageHeight']
        image['width'] = obj['imageWidth']
        image['id'] = self.img_id
        image['file_name'] = os.path.basename(path).replace(".json", ".jpg")
        return image

    # 构建COCO的annotation字段
    def _annotation(self, shape, img_id):
        label = shape['label']
        points = shape['points']
        annotation = {}
        if shape['shape_type'] != "circle" and shape['shape_type'] != "polygon":
            print(f'==================== ', shape['shape_type'], ' =====================')

        # TODO: to_coco与to_aug_coco关于点阵的处理位置不同
        #         if shape['shape_type'] == "circle":
        #             xy = [tuple(point) for point in points]
        #             (cx, cy), (px, py) = xy # 圆心和第二点
        #             # TODO：将圆形转换为多边形
        #             d = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
        #             d1 = math.sqrt((d ** 2) / 2)
        #             points = [[cx, cy-d], [cx-d1, cy-d1], [cx-d, cy], [cx-d1, cy+d1],
        #                      [cx, cy+d], [cx+d1, cy+d1], [cx+d, cy], [cx+d1, cy-d1]]

        annotation['id'] = self.ann_id
        annotation['image_id'] = img_id
        annotation['category_id'] = int(self.classname_to_id[label])
        annotation['segmentation'] = [np.asarray(points).flatten().tolist()]
        annotation['bbox'] = self._get_box(points)
        annotation['iscrowd'] = 0
        annotation['area'] = annotation['bbox'][2] * annotation['bbox'][3] # 这儿需要计算
        return annotation

    # 读取json文件，返回一个json对象
    def read_jsonfile(self, path):
        with open(path, "r", encoding='utf-8') as f:
            return json.load(f)

    # COCO的格式： [x1,y1,w,h] 对应COCO的bbox格式
    def _get_box(self, points):
        min_x = min_y = np.inf
        max_x = max_y = 0
        for x, y in points:
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
        return [int(np.floor(min_x)), int(np.floor(min_y)), int(np.ceil(max_x - min_x)), int(np.ceil(max_y - min_y))]
        #return [min_x, min_y, max_x - min_x, max_y - min_y]



# 解析 程序参数
def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--saved_coco_path', type=str, default="/home/zhangjw/research/skin_data/", help='saved_coco_path')
    parser.add_argument('--del_old_dir', type=bool, default=False, help='del_old_dir')
    parser.add_argument('--del_sub_old_dir', type=bool, default=True, help='del_sub_old_dir')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')

    args = parser.parse_args(
        # ['configs/skin/skin_config.py', '--work_dir', './work_dirs/skin_config_0831']
    )  #
    return args


# 程序主入口
def main():
    # 生成数据集列表
    gen_dataset_list = [
        {
            'split_type': 'train', 'crop_type': 'crop', 'crop_size': 1024, 'mask_flag':True
        },
        {
            'split_type': 'val', 'crop_type': 'crop', 'crop_size': 1024, 'mask_flag':True
        },
        # {
        #     'split_type': 'train', 'crop_type': 'whole',
        # },
        # {
        #     'split_type': 'val', 'crop_type': 'whole',
        # },
        # {
        #     'split_type': 'all', 'crop_type': 'whole',
        # },
    ]

    # 参数
    args = parse_args()

    # 保存coco根目录
    labelme_path = os.path.join(args.saved_coco_path, "labelme")

    # 是否删除旧的目录 annotations images
    # 创建存放标注文件的目录
    coco_annotation_dir = os.path.join(args.saved_coco_path, "coco1/annotations/")
    if args.del_old_dir:
        if os.path.exists(coco_annotation_dir):
            shutil.rmtree(coco_annotation_dir)
    if not os.path.exists(coco_annotation_dir):
        os.makedirs(coco_annotation_dir)

    # 创建存放图像文件的目录
    coco_img_dir = os.path.join(args.saved_coco_path, "coco1/images/")
    if args.del_old_dir:
        if os.path.exists(coco_img_dir):
            shutil.rmtree(coco_img_dir)
    if not os.path.exists(coco_img_dir):
        os.makedirs(coco_img_dir)


    # 获取labemel目录下所有的json文件列表
    all_path_list = glob.glob(os.path.join(labelme_path, "*.json"))

    # 生成图片名称列表用
    # img_name_list = [path.split('/')[-1].split('.')[0] for path in all_path_list]
    # print('img_name_list:', img_name_list)

    # 数据划分,这里没有区分val2017和tran2017目录，所有图片都放在images目录下
    # train_path_list, val_path_list = train_test_split(json_path_list, test_size=0.1)
    train_path_list, val_path_list = split_train_val_by_name(all_path_list)

    print('train_path_list:', len(train_path_list))
    print('train_path_list[0]:', train_path_list[0])
    print('val_path_list:', len(val_path_list))
    print('val_path_list[0]:', val_path_list[0])

    # 'split_type': 'train', 'crop_type': 'crop', 'crop_size': 1024,
    #'split_type': 'val', 'crop_type': 'crop', 'crop_size': 1024,
    #'split_type': 'train', 'crop_type': 'whole',
    for gen_dataset_json in gen_dataset_list:
        split_type = gen_dataset_json['split_type']
        crop_type = gen_dataset_json['crop_type']
        mask_flag = gen_dataset_json['mask_flag']
        crop_size = gen_dataset_json['crop_size'] if 'crop_size' in gen_dataset_json else None

        # 是否为crop
        if crop_size:
            coco_annotation_dir_dataset = os.path.join(coco_annotation_dir, '{}_{}_{}.json'.format(split_type, crop_type, crop_size))
            coco_img_dir_dataset = os.path.join(coco_img_dir, '{}_{}_{}'.format(split_type, crop_type, crop_size))
        else:
            coco_annotation_dir_dataset = os.path.join(coco_annotation_dir, '{}_{}.json'.format(split_type, crop_type))
            coco_img_dir_dataset = os.path.join(coco_img_dir, '{}_{}'.format(split_type, crop_type))

        # dataset图像数据文件夹
        if args.del_sub_old_dir:
            if os.path.exists(coco_img_dir_dataset):
                shutil.rmtree(coco_img_dir_dataset)
        if not os.path.exists(coco_img_dir_dataset):
            os.makedirs(coco_img_dir_dataset)

        # 把训练集转化为COCO的json格式
        print('begin to gendata: {} {} {}!'.format(split_type, crop_type, crop_size))

        l2c_train = Lableme2CoCo()
        # 获取训练标注数据
        if crop_type == 'crop':
            dataset_instance = l2c_train.to_aug_coco(eval('{}_path_list'.format(split_type)), coco_img_dir_dataset, mask_flag=mask_flag)
        else:
            dataset_instance = l2c_train.to_coco(eval('{}_path_list'.format(split_type)), coco_img_dir_dataset)

        # 保存coco标注文件
        l2c_train.save_coco_json(dataset_instance, coco_annotation_dir_dataset)



# 程序主入口
if __name__ == '__main__':
    main()
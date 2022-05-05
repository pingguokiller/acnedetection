from mmdet.datasets import PIPELINES
import numpy as np
import copy
import cv2
from matplotlib import pyplot as plt

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


@PIPELINES.register_module()
class MyTransform:
    def __init__(self, crop_size=1024):
        self.crop_size = crop_size
        self.useless_pad = 50
        self.count_array = np.load("/home/zhangjw/research/skin_data/distribution/skin_dis1.npy")
        #print(self.count_array.shape)
        # 与 opencv 图像坐标系一致

    # 直接线性比例回导致，样本多的图片采集很多，少的极少，概率相差100倍，这里我们用log函数舒缓比例，基本上100倍->8倍 几倍这个概率
    def log_curve(self, z, t=1):
        s = np.log(z / t + 1)
        p = s / np.sum(s)
        return p
    #
    def sample_by_dis(self):
        oned_count = self.count_array.flatten()
        index_list = range(oned_count.shape[0])

        p = self.log_curve(oned_count, 8)
        sampled_index = np.random.choice(index_list, 1, replace=True, p=p)

        # 一致性还原
        y_index = sampled_index % self.count_array.shape[1]
        x_index = sampled_index // self.count_array.shape[0]

        return x_index, y_index

    # 根据多边形点坐标置黑， return masked_img
    def draw_mask_by_polycorlist(self, img, cor_list):
        pts = np.array(cor_list, np.int32)  # 数据类型必须为 int32
        pts = pts.reshape((-1, 1, 2))

        cv2.polylines(img, np.int32([pts]), 1, 0)
        cv2.fillPoly(img, np.int32([pts]), 0)

        return img

    #
    def __call__(self, results):
        new_results = copy.deepcopy(results)

        # 所有的xy以图像cv坐标体现
        x_len = new_results['img_shape'][1] / self.count_array.shape[0]
        y_len = new_results['img_shape'][0] / self.count_array.shape[1]

        label_num = new_results['ann_info']['bboxes'].shape[0]
        old_bboxes = new_results['ann_info']['bboxes']
        old_masks = new_results['ann_info']['masks']
        old_labels = new_results['ann_info']['labels']

        # 如果没有采样到gt 需要重复采样
        while(1):
            # 采样位置
            x_index, y_index = self.sample_by_dis()
            x_center = x_index * x_len + 0.5 / x_len
            y_center = y_index * y_len + 0.5 / y_len
            leftup_x = x_center - 0.5/self.crop_size
            leftup_y = y_center - 0.5/self.crop_size

            # 限制范围
            leftup_x = int(np.clip(leftup_x, self.useless_pad, results['img_shape'][1] - self.crop_size - self.useless_pad))
            leftup_y = int(np.clip(leftup_y, self.useless_pad, results['img_shape'][0] - self.crop_size - self.useless_pad))

            # sample example
            # 保存图像, 图像的第一个纬度是行（高）
            sub_img_numpy = copy.deepcopy(results['img'][leftup_y:leftup_y + self.crop_size, leftup_x:leftup_x + self.crop_size, :])

            new_bboxes = []
            new_labels = []
            new_masks = []
            # 遍历所有的标注
            for label_index in range(label_num):
                tmp_bbox = copy.deepcopy(old_bboxes[label_index])

                # 检查是否重叠
                if doOverlap(Point(tmp_bbox[0], tmp_bbox[1]), Point(tmp_bbox[2], tmp_bbox[3]), Point(leftup_x, leftup_y),
                             Point(leftup_x + self.crop_size, leftup_y + self.crop_size)):
                    # 先计算该皮损的bbox
                    tmp_bbox[0] -= leftup_x
                    tmp_bbox[2] -= leftup_x
                    tmp_bbox[0] = np.clip(tmp_bbox[0], 1, self.crop_size - 1)
                    tmp_bbox[2] = np.clip(tmp_bbox[2], 1, self.crop_size - 1)
                    tmp_bbox[1] -= leftup_y
                    tmp_bbox[3] -= leftup_y
                    tmp_bbox[1] = np.clip(tmp_bbox[1], 1, self.crop_size - 1)
                    tmp_bbox[3] = np.clip(tmp_bbox[3], 1, self.crop_size - 1)

                    # 再计算所有的多边形点，不能用.copy(), 这个特直接用copy根本就没有复制子对象
                    tmp_points = np.array(old_masks[label_index][0]).reshape(-1, 2)

                    # 将大图坐标转换为小图坐标 point: (x, y) (水平坐标，垂直坐标)
                    bbox_split_flag = False
                    for point_index, point in enumerate(tmp_points):
                        point[0] -= leftup_x
                        point[1] -= leftup_y

                        # 标注的边框可能不能和图像边框重合, 这儿判断一个bbox是否被拆分
                        if point[0] < 1 or point[0] > (self.crop_size - 1) or point[1] < 1 or point[1] > (self.crop_size - 1):
                            bbox_split_flag = True

                        point[0] = np.clip(point[0], 1, self.crop_size - 1)  # x
                        point[1] = np.clip(point[1], 1, self.crop_size - 1)  # y

                        tmp_points[point_index] = point

                    # 该处标注不要且应该将原图该部分置为黑色
                    if bbox_split_flag:
                        # 画图
                        sub_img_numpy = self.draw_mask_by_polycorlist(sub_img_numpy, tmp_points)
                    else:
                        new_bboxes.append(tmp_bbox)
                        new_masks.append([tmp_points.flatten().tolist()])
                        new_labels.append(old_labels[label_index])

            if len(new_labels) > 0:
                new_results['ann_info']['labels'] = np.array(new_labels)
                new_results['ann_info']['bboxes'] = np.array(new_bboxes)
                new_results['ann_info']['masks'] = new_masks
                new_results['img'] = sub_img_numpy
                new_results['img_shape'] = sub_img_numpy.shape
                new_results['img_info']['height'] = sub_img_numpy.shape[0]
                new_results['img_info']['width'] = sub_img_numpy.shape[0]
                new_results['position'] = {
                    'x_index': x_index,
                    'x_len':self.count_array.shape[0],
                    'y_index': y_index,
                    'y_len':self.count_array.shape[1],
                }

                # for bbox in new_bboxes:
                #     cv2.rectangle(sub_img_numpy, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), thickness=3)
                #     sub_img_numpy = cv2.cvtColor(sub_img_numpy, cv2.COLOR_BGR2RGB)
                # plt.figure(figsize=(12, 9))
                # plt.imshow(sub_img_numpy)
                # plt.show()
                break

        return new_results

import numpy as np
import cv2
import os
from tqdm import tqdm

def cal_mean_std(imgs):
    if len(imgs.shape) == 2:
        channel_num = imgs.shape[1]
        mean_list, std_list = np.zeros((channel_num)), np.zeros((channel_num))
        for i in tqdm(range(channel_num)):
            mean_list[i] = imgs[:, i].mean()
            std_list[i] = imgs[:, i].std()
    else:
        mean_list, std_list = np.zeros((1)), np.zeros((1))
        mean_list[0] = imgs[:, :, :].mean()
        std_list[0] = imgs[:, :, :].std()

    return mean_list, std_list

data_root = '/home/zhangjw/research/skin_data/coco/'
imgs_path = os.path.join(data_root, 'images/train2017/')
imgs_path_list = os.listdir(imgs_path)


img_h, img_w = 1024, 1024  # 根据自己数据集适当调整，影响不大
img_list = []

for item in tqdm(imgs_path_list[:]):
    img = cv2.imread(os.path.join(imgs_path ,item)) # BGR
    if img.shape[0] != img_w:
        img = cv2.resize(img, (img_w, img_h))

    img = img.reshape(-1, 3)

    #one_img_mean = img.mean(axis)
    #img = img[:, :, :, np.newaxis]
    #print('img.shape1:', img.shape)
    img_list.append(img)

imgs = np.array(img_list)
imgs = imgs.reshape(-1, 3)
#imgs = np.concatenate(img_list, axis=3)
# TODO: 注意图像是否除以255
print(imgs.shape)

train_mean, train_std = cal_mean_std(imgs)
print('train_mean, train_std:', train_mean, train_std)


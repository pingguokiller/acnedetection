"""
【用于直接导入图像（非接口上传）的名称处理脚本】
处理当前文件夹的其他图像文件的文件名，更改后查看是否在原文件夹中出现，若出现则进行更改

params:
 - last_path: 原文件夹位置
 - path2: 当前文件夹位置
"""

import os


# 规则化图像文件名字
def namenormal(name):
    name = name.replace(' ', '_')
    name = name.replace(',', '_')
    name = name.replace('-', '_')
    name = name.replace('—', '')
    name = name.replace('°', '')
    name = name.replace('、', '')
    name = name.replace('，', '')
    name = name.replace('？', '')
    name = name.replace('=', '')
    name = name.replace(')', '')
    name = name.replace('(', '')
    name = name.upper()
    name = name.split('.')[0] + '.jpg'
    return name


# 将原文件重新命名为信息的文件名称
def rename_func(img_path_list, path):
    for file_name in os.listdir(path):

        if (file_name.endswith('.JPG') or file_name.endswith('.jpg')):
            new_file_name = namenormal(file_name)

            if (new_file_name != file_name):
                print("Old:", file_name, "New:", new_file_name)
                os.rename(path + '/' + file_name, path + '/' + new_file_name)


#
if __name__ == '__main__':
    #
    origin_img_dir = 'F:/skin_data/raw'
    image_path_list = os.listdir(origin_img_dir)

    rename_func(image_path_list, origin_img_dir)

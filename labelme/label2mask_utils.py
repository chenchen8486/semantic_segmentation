#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import numpy as np
import cv2
import json
import time
import shutil


def list_images(path, file_type='images'):
    """
    描述：列出文件夹中所有的文件，然后返回文件夹内所有文件的全路径列表
    :param path:包含了文件的文件夹路径
    :param file_type: images or any
    :return: 文件夹内所有文件的全路径名称的列表
    """
    image_format_list = ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp']
    paths_list = []
    for file_and_dir in os.listdir(path):  # 遍历给定文件夹路径下的所有文件名称
        if os.path.isfile(os.path.join(path, file_and_dir)):  # 判断当前路径下的指定文件是否存在。
            if file_type == 'images':
                if os.path.splitext(file_and_dir)[1] in image_format_list:  # 如果文件名称的扩展名存在于指定的列表内
                    paths_list.append(os.path.abspath(os.path.join(path, file_and_dir)))
            elif file_type == 'any':
                paths_list.append(os.path.abspath(os.path.join(path, file_and_dir)))
            else:
                if os.path.splitext(file_and_dir)[1] == file_type:
                    paths_list.append(os.path.abspath(os.path.join(path, file_and_dir)))
    return paths_list


def cut_same_json_and_image(data_path, save_path):
    '''
    功能： 在标注的文件夹内，往往有些数据标注了存在json文件，而有些文件没有json，
    本函数功能就是将相同名称的json和图像都剪切到指定文件夹，仅把剩余单独的图像留在原处。
    :param data_path: 存有图像和Json的文件夹数据
    :param save_path: 指定的剪切文件夹路径
    :return:
    '''
    for root, dirs, files in os.walk(data_path):
        for file in files:
            extension_name = os.path.splitext(file)[1]
            if extension_name == '.json':
                img_base_name = os.path.splitext(file)[0]

                src_img_path = root + '/' + img_base_name + '.bmp'
                dst_img_path = save_path + '/' + img_base_name + '.bmp'
                shutil.move(src_img_path, dst_img_path)

                src_json_path = root + '/' + file
                dst_json_path = save_path + '/' + file
                shutil.move(src_json_path, dst_json_path)
            print(file)


def delete_all_file(path):
    """
    删除某一目录下的所有文件或文件夹
    :param path:
    :return:
    """
    del_list = os.listdir(path)
    for f in del_list:
        file_path = os.path.join(path, f)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    print("Delete all file finished ...")


def create_all_need_label_file(generate_label_path):
    """
    功能：创建ground truth所需要的所有文件夹
    :param generate_label_path: 生成的ground truth的根目录，所有文件夹都在根目录内创建
    :return:
    """
    """创建生成的最终labels的文件夹"""
    label_path = os.path.join(generate_label_path, 'labels')
    if not os.path.exists(label_path):
        os.makedirs(label_path)

    """创建ok and ng样本的两个文件夹（所有标注文件中有些没有被标注，即没有json，那么这些图像就存到ok，否则就存到ng）"""
    ok_path = os.path.join(generate_label_path, 'ok')
    if not os.path.exists(ok_path):
        os.makedirs(ok_path)
    ng_path = os.path.join(generate_label_path, 'ng')
    if not os.path.exists(ng_path):
        os.makedirs(ng_path)

    """crop小图"""
    crop_path = os.path.join(generate_label_path, 'croped_image')
    if not os.path.exists(crop_path):
        os.makedirs(crop_path)
    print('Create all file finished ...')
    return label_path, ok_path, ng_path, crop_path
#!/usr/bin/env python2
# coding: utf-8
from __future__ import print_function
import os, cv2, sys, math, json, time, datetime, random
import tensorflow as tf
import numpy as np

''' 参数配置 '''
config_path = 'D:/3_deep_leaerning_project/semantic_segmentation/config_train_tatal.json'
config_data = json.loads(open(config_path).read(), encoding='utf-8')
IMAGE_SIZE = config_data['IMAGE_SIZE']
num_classes = config_data['NUM_CLASSES'] # 类别数量，多分类需要背景类
CHANNEL_NUM = config_data['CHANNEL_NUM']  # 'rgb' or 'gray'


def rotate(image, angle=90):
    height, width = image.shape[:2]
    degree = angle
    heightNew = int(width * math.fabs(math.sin(math.radians(degree))) +
                    height * math.fabs(math.cos(math.radians(degree))))
    widthNew = int(height * math.fabs(math.sin(math.radians(degree))) +
                   width * math.fabs(math.cos(math.radians(degree))))
    matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)
    matRotation[0, 2] += (widthNew - width) / 2
    matRotation[1, 2] += (heightNew - height) / 2
    imgRotation = cv2.warpAffine(image, matRotation, (widthNew, heightNew), borderValue=(0, 0, 0))
    return imgRotation


def resize_image(image, max_edge=IMAGE_SIZE, interpolation='INTER_LINEAR'):
    # 如果最大边长大于max_edge，限制最大边长到max_edge
    max_size = image.shape[0] if image.shape[0] > image.shape[1] else image.shape[1]
    if max_size > max_edge:
        resize_ratio = max_edge / float(max_size)
        if interpolation == 'INTER_LINEAR':
            image = cv2.resize(image, None, fx=resize_ratio, fy=resize_ratio, interpolation=cv2.INTER_LINEAR)
        else:
            image = cv2.resize(image, None, fx=resize_ratio, fy=resize_ratio, interpolation=cv2.INTER_NEAREST)
    return image, None


def mean_image_subtraction(images, means=[123.68, 116.78, 103.94]):
    '''
    image normalization
    :param images:
    :param means:
    :return:
    '''
    num_channels = images.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')
    channels = tf.split(axis=3, num_or_size_splits=num_channels, value=images)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=3, values=channels)


def random_rotate_and_flip(image, labels, rand_threhold=0.3):
    """
    image: 原图
    labels: list,根据分类数量可能有多个label
    """
    # 随机旋转
    rand_val = random.random()
    if rand_val < rand_threhold:
        angle = random.choice([90, 180, 270])
        image = rotate(image, angle=angle)
        labels = [rotate(label, angle=angle) for label in labels]

    # 随机水平翻转
    if random.randint(0, 1):
        image = cv2.flip(image, 1)
        labels = [cv2.flip(label, 1) for label in labels]

    # 随机垂直翻转
    if random.randint(0, 1):
        image = cv2.flip(image, 0)
        labels = [cv2.flip(label, 0) for label in labels]

    # 二值化，确保label是二值的
    thre_labels = []
    for label in labels:
        _, thre_label = cv2.threshold(label, 128, 255, cv2.THRESH_BINARY)
        thre_labels.append(thre_label)

    return image, thre_labels


def random_stretch(image, labels, rand_threhold=0.3):
    rand_val = random.random()
    if rand_val < rand_threhold:
        # 随机拉伸
        stretch_ratio = random.uniform(0.75, 1.25)
        rand_w = image.shape[1]
        rand_h = image.shape[0]
        if random.randint(0, 1):
            rand_w = int(rand_w * stretch_ratio)
        else:
            rand_h = int(rand_h * stretch_ratio)

        image = cv2.resize(image, (rand_w, rand_h))
        labels = [cv2.resize(label, (rand_w, rand_h)) for label in labels]

    return image, labels


def random_resize_and_paste(image, labels, rand_threhold=0.25, corner_threhold=0.7):
    rand_val = random.random()
    if rand_val < rand_threhold:
        # 对样本进行一定概率的随机尺寸缩放
        rand_w = np.random.randint(int(IMAGE_SIZE * 0.7), IMAGE_SIZE)
        rand_h = np.random.randint(int(IMAGE_SIZE * 0.7), IMAGE_SIZE)
        image = cv2.resize(image, (rand_w, rand_h))
        labels = [cv2.resize(label, (rand_w, rand_h)) for label in labels]

    h = image.shape[0]
    w = image.shape[1]

    # 计算图像粘贴的起始点范围
    range_x = IMAGE_SIZE - w - 1
    range_y = IMAGE_SIZE - h - 1

    if range_x < 0:
        range_x = 0
    if range_y < 0:
        range_y = 0

    # 随机生成起始点
    rand_x = random.randint(0, range_x)
    rand_y = random.randint(0, range_y)

    # 按照一定的概率比例，将训练数据粘贴到空白图像的4个角
    pos_list = [(0, 0), (0, range_x), (range_y, 0), (range_y, range_x)]
    rand_index = random.randint(0, 3)
    rand_pb = random.random()  # 随机产生一个概率
    if rand_pb < corner_threhold:
        rand_y = pos_list[rand_index][0]
        rand_x = pos_list[rand_index][1]

    # 生成空白图像（全白色），用于装载原始图像
    if CHANNEL_NUM == 3:
        paste_image = np.ones([IMAGE_SIZE, IMAGE_SIZE, 3], dtype=np.uint8) * 255
        paste_image[rand_y:h + rand_y, rand_x:w + rand_x, :] = image[:, :, :]
    else:
        paste_image = np.ones([IMAGE_SIZE, IMAGE_SIZE], dtype=np.uint8) * 255
        paste_image[rand_y:h + rand_y, rand_x:w + rand_x] = image[:, :]

    new_labels = []
    for n in range(num_classes):
        # 生成空白图像（全黑色），用于装载前景掩模图像
        score_map = np.zeros([IMAGE_SIZE, IMAGE_SIZE], dtype=np.uint8)
        score_map[rand_y:h + rand_y, rand_x:w + rand_x] = labels[n][:, :]
        new_labels.append(score_map)

    return paste_image, new_labels


def random_jpg_quality(input_image):
    # 压缩原图
    if random.randint(0, 1):
        jpg_quality = np.random.randint(60, 100)
        _, input_image = cv2.imencode('.jpg', input_image, [1, jpg_quality])
        if CHANNEL_NUM == 3:
            input_image = cv2.imdecode(input_image, 1)
        else:
            input_image = cv2.imdecode(input_image, 0)

    return input_image


def random_bright(input_image):
    # 随机亮度
    if random.randint(0, 1):
        factor = 1.0 + random.uniform(-0.4, 0.4)
        table = np.array([min((i * factor), 255.) for i in np.arange(0, 256)]).astype(np.uint8)
        input_image = cv2.LUT(input_image, table)

    return input_image


def random_paste_croped_image(crop_image_list, image, label, crop_rand_threhold=0.3):
    rand_val = random.random()
    if len(crop_image_list) == 0:
        return image, label

    if rand_val < crop_rand_threhold:
        crop_image_path = random.choice(crop_image_list)
        crop_image = cv2.imread(crop_image_path)

        # 随机旋转
        if random.randint(0, 1):
            angle = random.choice([90, 180, 270])
            crop_image = rotate(crop_image, angle=angle)

        # 随机水平翻转
        if random.randint(0, 1):
            crop_image = cv2.flip(crop_image, 1)

        # 随机垂直翻转
        if random.randint(0, 1):
            crop_image = cv2.flip(crop_image, 0)

        h = crop_image.shape[0]
        w = crop_image.shape[1]

        h_start = np.random.randint(1, IMAGE_SIZE-h-1)
        w_start = np.random.randint(1, IMAGE_SIZE-w-1)

        image[h_start: h_start+h, w_start: w_start+w, :] = crop_image[:]

        label[h_start: h_start+h, w_start: w_start+w] = 255

    return image, label
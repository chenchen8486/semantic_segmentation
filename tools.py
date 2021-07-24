#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import numpy as np
import cv2
import time
import json
import tensorflow as tf
from builders import model_builder


''' 参数配置 '''
config_path = 'D:/3_deep_leaerning_project/semantic_segmentation/config_inference.json'
config_data = json.loads(open(config_path).read(), encoding='utf-8')
NUM_CLASSES = config_data['NUM_CLASSES']
SUB_IMAGE_SIZE = config_data['SUB_IMAGE_SIZE']
MODEL_NAME = config_data['MODEL_NAME']
FRONTEND = config_data['FRONTEND']
CHECKPOINT_DIR = config_data['CHECKPOINT_DIR']
SCORE_MAP_THRESH = config_data['SCORE_MAP_THRESH']
BATCH_SIZE = config_data['BATCH_SIZE']
MERGE_DIS_THRE = config_data['MERGE_DIS_THRE']


# NUM_CLASSES = 5
# SUB_IMAGE_SIZE = 768
# MODEL_NAME = "DeepLabV3_plus"
# FRONTEND = "MobileNetV2"
# CHECKPOINT_DIR = "D:/3_deep_leaerning_project/semantic_segmentation_ckpt"
# SCORE_MAP_THRESH = 128
# BATCH_SIZE = 3
# MERGE_DIS_THRE = 30



def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

class SemanticSegment(object):

    def __init__(self):
        config = tf.ConfigProto()

        # 使用allow_growth option，刚一开始分配少量的GPU容量，然后按需慢慢的增加，由于不会释放
        # 内存，所以会导致碎片
        config.gpu_options.allow_growth = True
        # config.gpu_options.per_process_gpu_memory_fraction = GPU_MEMORY_FRACTION
        self.num_classes = NUM_CLASSES
        tf.reset_default_graph()
        self.session = tf.Session(config=config)
        self.session.run(tf.global_variables_initializer())
        with self.session.as_default():
            self.input_image = tf.placeholder(tf.float32, shape=[None, SUB_IMAGE_SIZE, SUB_IMAGE_SIZE, 3])

            self.prediction, _ = model_builder.build_model(model_name=MODEL_NAME,
                                                           frontend=FRONTEND,
                                                           net_input=self.input_image,
                                                           num_classes=NUM_CLASSES,
                                                           crop_width=SUB_IMAGE_SIZE,
                                                           crop_height=SUB_IMAGE_SIZE)
            saver = tf.train.Saver()
            ckpt = tf.train.latest_checkpoint(CHECKPOINT_DIR)
            if ckpt is not None:
                saver.restore(self.session, ckpt)
            zero_image = np.zeros([1, SUB_IMAGE_SIZE, SUB_IMAGE_SIZE, 3], dtype=np.uint8)
            self.session.run(self.prediction, feed_dict={self.input_image: zero_image})
            # ALL_LOG_OBJ.logger.info('Init model finished!')

    def predict_multi_label(self, image, label_name_list):
        im_in = image
        im_in = np.float32(im_in) / 255.0
        if len(im_in.shape) < 3:
            im_in = np.expand_dims(im_in, axis=2)

        prediction = self.session.run(self.prediction, feed_dict={self.input_image: im_in})
        prediction = sigmoid(prediction)
        batch_result = []
        for pred_id, pred_result in enumerate(prediction):
            each_prediction_result = {}
            for class_id in range(self.num_classes):  # 有多少个类别,就循环多少次进行存储, 每个图像存储成一个字典，
                # 获取score map(单张图像的每个类别的score map)
                pred = (pred_result[:, :, class_id] * 255).astype(np.uint8)
                # cv2.imwrite("C:/Users/admin/Desktop/test/aaa.bmp", pred)
                # cv2.imshow("test", pred)
                # cv2.waitKey()
                # 获取pred_mask(单张图像的每个类别的pred_mask)
                _, pred_mask = cv2.threshold(pred, SCORE_MAP_THRESH, 255, cv2.THRESH_BINARY)
                # 结果数据存储
                each_prediction_result[label_name_list[class_id]] = pred_mask.astype(np.uint8)
            batch_result.append(each_prediction_result)  # 最后把字典存到List中
        return batch_result


def create_batch_list(crop_image_list):
    batch_list = []
    temp = []
    for crop_id, crop_img in enumerate(crop_image_list):
        temp.append(crop_img)
        if len(temp) == BATCH_SIZE:
            batch_list.append(temp)
            temp = []
    if len(temp) != 0:
        batch_list.append(temp)
    return batch_list


def crop_right_camera_overlap_image(input_image, batch_w, batch_h, overlap_x=256, overlap_y=50):
    height, width = input_image.shape[0], input_image.shape[1]
    crop_rect_list = []
    row_id, col_id = 0, 0
    cut_img_w_th = int(batch_w / 3)
    while (row_id + batch_h) <= height:
        if row_id == 0:
            batch_start_y = 0
        else:
            batch_start_y = batch_start_y + batch_h - overlap_y

        while (col_id + batch_w) <= width:
            if col_id == 0:
                batch_start_x = 0
            else:
                batch_start_x = batch_start_x + batch_w - overlap_x
            batch = (batch_start_x, batch_start_y, batch_w, batch_h)
            crop_rect_list.append(batch)
            col_id = col_id + batch_w - overlap_x
        if abs(width - col_id) > cut_img_w_th:  # 如果当前列
            batch_start_x = width - batch_w
            batch = (batch_start_x, batch_start_y, batch_w, batch_h)
            crop_rect_list.append(batch)
            row_id = row_id + batch_h - overlap_y
            col_id = 0
        else:
            row_id = row_id + batch_h - overlap_y
            col_id = 0

    # 为了对最后一行进行crop的补救
    while (col_id + batch_w) <= width:
        if col_id == 0:
            batch_start_x = 0
        else:
            batch_start_x = batch_start_x + batch_w - overlap_x
        batch_start_y = height - batch_h
        batch = (batch_start_x, batch_start_y, batch_w, batch_h)
        crop_rect_list.append(batch)
        col_id = col_id + batch_w - overlap_x
    if abs(width - col_id) > cut_img_w_th:
        batch_start_x = width - batch_w
        batch_start_y = height - batch_h
        batch = (batch_start_x, batch_start_y, batch_w, batch_h)
        crop_rect_list.append(batch)

    return crop_rect_list


def crop_overlap_image(input_image, direction_type, batch_w, batch_h, overlap_x=0, overlap_y=0):
    crop_image_list = []
    if direction_type == 'right':
        crop_rect_list = crop_right_camera_overlap_image(input_image, batch_w, batch_h, overlap_x, overlap_y)
    else:
        cpy_input_image = np.copy(input_image)
        flip_input_image = np.flip(cpy_input_image, 0)
        crop_rect_list = crop_right_camera_overlap_image(flip_input_image, batch_w, batch_h, overlap_x, overlap_y)
        crop_rect_list = [[flip_input_image.shape[1] - rr[0] - rr[2], rr[1], rr[2], rr[3]] for rr in crop_rect_list]

    for batch_id, rect in enumerate(crop_rect_list):
        start_x = rect[0]
        start_y = rect[1]
        end_x = rect[0] + rect[2]
        end_y = rect[1] + rect[3]
        crop_image = input_image[start_y:end_y, start_x:end_x]
        crop_image = cv2.cvtColor(crop_image, cv2.COLOR_GRAY2BGR)
        crop_image_list.append(crop_image)
    return crop_rect_list, crop_image_list


def direction_classify(input_image):
    resize_image = cv2.resize(input_image, (0, 0), fx=0.2, fy=0.2, interpolation=cv2.INTER_LINEAR)
    first_col = resize_image[:, 0]
    last_col = resize_image[:, -1]
    first_col_std = np.std(first_col, ddof=1)
    last_col_std = np.std(last_col, ddof=1)

    if first_col_std <= last_col_std:
        direction_type = 'left'
    else:
        direction_type = 'right'
    return direction_type


def image_stitiching(crop_img_list, crop_rect_list, image_height, image_width):
    # 设置整图像buffer
    stitching_image = np.zeros((image_height, image_width), np.uint8)

    for id, rect in enumerate(crop_rect_list):
        if crop_img_list[id].ndim == 3:
            crop_img_list[id] = cv2.cvtColor(crop_img_list[id], cv2.COLOR_BGR2GRAY)
        left = rect[0]
        top = rect[1]
        right = rect[0] + rect[2]
        bottom = rect[1] + rect[3]
        stitching_crop_img = stitching_image[top:bottom, left:right]
        merge_crop_img = np.bitwise_or(stitching_crop_img, crop_img_list[id])
        stitching_image[top:bottom, left:right] = merge_crop_img
    return stitching_image


def is_overlap(rect1, rect2):
    '''
    计算两个矩形的交并比
    :param rect1:第一个矩形框。表示为x,y,w,h，其中x,y表示矩形右上角的坐标
    :param rect2:第二个矩形框。
    :return:是否有重叠部分
    '''
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    inter_w = (w1 + w2) - (max(x1 + w1, x2 + w2) - min(x1, x2))
    inter_h = (h1 + h2) - (max(y1 + h1, y2 + h2) - min(y1, y2))

    if inter_h <= 0 or inter_w <= 0:  # 代表相交区域面积为0
        return False
    else:
        return True


def need_merge(rect1, rect2, thre=MERGE_DIS_THRE):
    '''
    根据两个框的曼哈顿距离判断是否需要合并
    :param rect1:第一个矩形框。表示为x,y,w,h，其中x,y表示矩形右上角的坐标
    :param rect2:第二个矩形框。
    :return:是否需要合并
    '''
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2

    inter_w = (w1 + w2) - (max(x1 + w1, x2 + w2) - min(x1, x2))
    inter_h = (h1 + h2) - (max(y1 + h1, y2 + h2) - min(y1, y2))

    v_dis = abs(min((y1 + h1), (y2 + h2)) - max(y1, y2))
    h_dis = abs(min((x1 + w1), (x2 + w2)) - max(x1, x2))
    if v_dis + h_dis < thre or (v_dis < 0.5 * thre and inter_w > 0) or \
            (h_dis < 0.5 * thre and inter_h > 0) or is_overlap(rect1, rect2):
        return True
    elif (x1 > x2) and ((x1+w1) < (x2+w2)) and (y1 > y2) and ((y1+h1) < (y2+h2)):
        # rect1 在 rect2 内部
        return True
    elif (x2 > x1) and ((x2+w2) < (x1+w1)) and (y2 > y1) and ((y2+h2) < (y1+h1)):
        # rect2 在 rect1 内部
        return True
    else:
        return False
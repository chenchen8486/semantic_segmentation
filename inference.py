#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import numpy as np
import cv2
import time
import json
import tensorflow as tf
from builders import model_builder
from tools import *
from skimage.measure import label, regionprops
from tools import is_overlap, need_merge
from utils.utils import *

# 定义分割类的实例
semantic_segment_instance = SemanticSegment()

''' 参数配置 '''
config_path = 'D:/3_deep_leaerning_project/semantic_segmentation/config_inference.json'
config_data = json.loads(open(config_path).read(), encoding='utf-8')
SUB_IMAGE_SIZE = config_data['SUB_IMAGE_SIZE']
INFERENCE_ALL_IMAGE = config_data['INFERENCE_ALL_IMAGE']
os.environ['CUDA_VISIBLE_DEVICES'] = config_data['CUDA_VISIBLE_DEVICES']



def prediction(crop_image_list, label_name_list):
    # 把所有crop小图分组成batch进行存储
    batch_list = create_batch_list(crop_image_list)

    # 定义不同cls的列表，用来存储每类别的所有crop图像
    total_prediction_list = []
    for label_id in range(len(label_name_list)):
        total_prediction_list.append([])

    for batch_id, batch in enumerate(batch_list):  # 一张大图中所有的crop小图打包成batch，循环每个batch
        batch_result = semantic_segment_instance.predict_multi_label(batch, label_name_list)
        # 获取每张预测小图的不同类别的mask，存入各自cls的列表中
        for img_id in range(len(batch_result)):   # 循环一个batch里面的图片数量，比如batch=2，就循环2次，每次循环抓取所有的pred结果
            for label_id, label_name in enumerate(label_name_list):
                total_prediction_list[label_id].append(batch_result[img_id][label_name])

    return total_prediction_list


def draw_result_rect(image_ori, total_prediction_list, label_name_list, crop_rect_list, height, width, INFERENCE_ALL_IMAGE):
    label_list = []
    for label_id in range(len(label_name_list)):
        if INFERENCE_ALL_IMAGE:
            image_temp = image_stitiching(total_prediction_list[label_id], crop_rect_list, height, width)
            label_list.append(image_temp)
        else:
            label_list.append(total_prediction_list[label_id][0])

    cc_info = []
    # 计算连通域的属性
    for n, cc_label in enumerate(label_list):
        label_img = label(cc_label, neighbors=8, connectivity=2)
        all_connect_info = regionprops(label_img)

        for element in all_connect_info:
            if element.filled_area > 10:
                info = {}
                ly, lx, ry, rx = element.bbox
                info['box_left'] = lx
                info['box_top'] = ly
                info['box_right'] = rx
                info['box_bottom'] = ry
                info['box_width'] = info['box_right'] - info['box_left']
                info['box_height'] = info['box_bottom'] - info['box_top']
                info['box_area'] = info['box_width'] * info['box_height']
                info['type'] = n  # 缺陷类别
                info['merge_flag'] = False  # 是否被合并的标记，初始化为False
                cc_info.append(info)
    # 合并同类型并且有交叠的框
    for n in range(len(cc_info)):
        for k in range(len(cc_info)):
            if n != k and cc_info[n]['type'] == cc_info[k]['type'] \
                    and (cc_info[n]['merge_flag'] is False) \
                    and (cc_info[k]['merge_flag'] is False):
                bbox1 = (cc_info[n]['box_left'], cc_info[n]['box_top'],
                         cc_info[n]['box_right'] - cc_info[n]['box_left'],
                         cc_info[n]['box_bottom'] - cc_info[n]['box_top'])
                bbox2 = (cc_info[k]['box_left'], cc_info[k]['box_top'],
                         cc_info[k]['box_right'] - cc_info[k]['box_left'],
                         cc_info[k]['box_bottom'] - cc_info[k]['box_top'])

                if need_merge(bbox1, bbox2):
                    new_x = min(bbox1[0], bbox2[0])
                    new_y = min(bbox1[1], bbox2[1])
                    new_r = max(bbox1[0] + bbox1[2], bbox2[0] + bbox2[2])
                    new_b = max(bbox1[1] + bbox1[3], bbox2[1] + bbox2[3])
                    cc_info[n]['box_left'] = new_x
                    cc_info[n]['box_top'] = new_y
                    cc_info[n]['box_right'] = new_r
                    cc_info[n]['box_bottom'] = new_b
                    cc_info[k]['merge_flag'] = True
    if len(cc_info) > 0:
        cc_info = [cell for cell in cc_info if cell['merge_flag'] is False]

    image_bgr = image_ori.copy()
    if len(cc_info) != 0:
        for cc in cc_info:
            cv2.rectangle(image_bgr, (cc['box_left'], cc['box_top']), (cc['box_right'], cc['box_bottom']), (0, 0, 255), 1)
            for label_id in range(len(label_name_list)):
                if cc['type'] == label_id:
                    cv2.putText(image_bgr, label_name_list[label_id], (cc['box_left'],  cc['box_top']), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return image_bgr


def draw_result_mask(image_bgr, total_prediction_list):
    alpha = 0.95
    output_img = image_bgr.copy()

    color_list = [(0, 0, 255), (0, 255, 0), (255, 0, 0),
                  (255, 255, 0), (128, 128, 0), (0, 0, 128),
                  (0, 128, 128)]

    for inf_id, inf_mask in enumerate(total_prediction_list):
        inf_mask = inf_mask[0]
        inf_mask_bgr = cv2.cvtColor(inf_mask, cv2.COLOR_GRAY2BGR)
        white_pos = np.vstack(np.where(inf_mask != 0))
        inf_mask_bgr[white_pos[0][:], white_pos[1][:], :] = color_list[inf_id]
        output_img = output_img * alpha + inf_mask_bgr * (1-alpha)
    return output_img


if __name__ == '__main__':
    label_name_list = ['frontier', 'mark', 'corner', 'broken', 'chipping']
    # label_name_list = ['frontier', 'mark', 'corner',  'corner_dirty', 'dirty_pt', 'crack', 'bubble']
    image_file_path = "C:/Users/admin/Desktop/dt/defect/"
    save_result_path = "C:/Users/admin/Desktop/dt/result/"
    image_file_path_list = list_images(image_file_path)
    for name_id, image_path in enumerate(image_file_path_list):
        print(image_path)
        # 加载图像（灰度图像）
        image_file_name = os.path.join(image_file_path, image_path)
        image_ori = cv2.imread(image_file_name, -1)
        height, width = image_ori.shape[0], image_ori.shape[1]

        # 获取BGR图像
        if image_ori.ndim != 3:
            image_bgr = cv2.cvtColor(image_ori, cv2.COLOR_GRAY2BGR)
        else:
            image_bgr = image_ori.copy()

        # 判断当前inference是全图还是小图
        if INFERENCE_ALL_IMAGE:
            # get camera direction...
            direction_type = direction_classify(image_ori)
            # crop image...
            crop_rect_list, crop_image_list = crop_overlap_image(image_ori, direction_type, SUB_IMAGE_SIZE, SUB_IMAGE_SIZE)
            # inference...
            total_prediction_list = prediction(crop_image_list, label_name_list)
            # draw...
            draw_result_rect(image_bgr, total_prediction_list, label_name_list, crop_rect_list, height, width, INFERENCE_ALL_IMAGE)
        else:
            crop_image_list = [image_bgr]
            total_prediction_list = prediction(crop_image_list, label_name_list)
            # draw rect...
            result_img = draw_result_rect(image_bgr, total_prediction_list, label_name_list, [], 0, 0, INFERENCE_ALL_IMAGE)

            # draw mask...
            result_mask_with_img = draw_result_mask(image_bgr, total_prediction_list)
            # # 存储
            base_name = os.path.basename(image_path)
            split_name = os.path.splitext(base_name)
            image_name = split_name[0]
            cv2.imwrite(save_result_path + image_name + "_defect_.bmp", result_img)
            cv2.imwrite(save_result_path + image_name + "_defect_mask.bmp", result_mask_with_img)


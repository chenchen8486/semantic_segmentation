#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import cv2
import tensorflow as tf
import numpy as np
from builders import model_builder
# import model
import time
import shutil
import json
import copy
import argparse
import yaml
from utils.utils import COLOUR_LIST
from skimage.measure import label, regionprops

f = open(r'D:/3_deep_leaerning_project/semantic_segmentation/labelme/config_labelme.yaml', 'r', encoding='utf-8')
result = f.read()
yaml_data_dict = yaml.load(result, Loader=yaml.FullLoader)



GPU_MEMORY_FRACTION = yaml_data_dict['gpu_memory_fraction']
os.environ['CUDA_VISIBLE_DEVICES'] = yaml_data_dict['cuda_visible_devices']
num_classes = yaml_data_dict['num_classes']
model_name = yaml_data_dict['model_name']
frontend = yaml_data_dict['frontend']
ckpt_dir = yaml_data_dict['ckpt_dir']
IMAGE_WIDTH = yaml_data_dict['image_width']
IMAGE_HEIGHT = yaml_data_dict['image_height']
load_image_mode = cv2.IMREAD_COLOR  # IMREAD_COLOR, IMREAD_GRAYSCALE
class_model = yaml_data_dict['class_model']  # 配置多分类和多标签：multi_classify、multi_label

filled_area_thre = yaml_data_dict['filled_area_thre']
mask_seq = [yaml_data_dict['mask_seq']]

root_dir = yaml_data_dict['root_dir']
save_dir = yaml_data_dict['save_dir']

save_json = yaml_data_dict['save_json']
label_name = yaml_data_dict['label_name']
json_path = yaml_data_dict['json_path']


def order_points(pts):
    xSorted = pts[np.argsort(pts[:, 0]), :]  # 对x排序
    change = False
    # 如果右边的两个点的y值都比左边的小或者大,表明文本是竖着的且倾斜很厉害
    if (xSorted[2][1] < xSorted[0][1] and xSorted[2][1] < xSorted[1][1] \
            and xSorted[3][1] < xSorted[0][1] and xSorted[3][1] < xSorted[1][1]) or \
            (xSorted[2][1] > xSorted[0][1] and xSorted[2][1] > xSorted[1][1]
            and xSorted[3][1] > xSorted[0][1] and xSorted[3][1] > xSorted[1][1]):
        ySorted = pts[np.argsort(pts[:, 1]), :]  # 对x排序
        if abs(ySorted[2][1] - ySorted[1][1]) > abs(xSorted[2][0] - xSorted[1][0]):
            # 判断文本是竖着的
            change = True
    # xSorted[:, [1, 0]] = xSorted[:, [0, 1]]  # x,y对换
    leftMost = xSorted[:2, :]  # 前两名分到左边
    rightMost = xSorted[2:, :]  # 后两名分到右边
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]  # 对左边两个点的y进行排序
    # if leftMost[0][1] == leftMost[1][1]:
    if change:
        leftMost = leftMost[np.argsort(leftMost[:, 0]), :]  # 对x排序
        (bl, tl) = leftMost
    else:
        (tl, bl) = leftMost
    rightMost = rightMost[np.argsort(rightMost[:, 1]), :]  # 对右边两个点的y进行排序

    # if rightMost[0][1] == rightMost[1][1]:
    if change:
        rightMost = rightMost[np.argsort(rightMost[:, 0]), :]  # 对x排序
        (br, tr) = rightMost
    else:
        (tr, br) = rightMost
    return np.array([tl, tr, br, bl], dtype="int")


def generate_miniled_json(image_name, order_pts, label_list, imageWidth, imageHeight, save_path):
    bboxes = []
    for i, pts in enumerate(order_pts):
        if len(pts) == 2:
            box = {"group_id": None,
                   "shape_type": "rectangle",
                   "points": pts,
                   "flags": {},
                   "label": label_list[i]}
        else:
            box = {"group_id": None,
                   "shape_type": "polygon",
                   "points": pts,
                   "flags": {},
                   "label": label_list[i]}
        bboxes.append(box)
    json_template = {"imagePath": image_name,
                     "imageData": None,
                     "shapes": bboxes,
                     "version": "4.5.6",
                     "flags": {},
                     "imageWidth": imageWidth,
                     "imageHeight": imageHeight
                     }
    json_name = os.path.splitext(image_name)[0] + '.json'
    with open(os.path.join(save_path, json_name), 'w') as f:
        json.dump(json_template, f, indent=4)


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis=0)


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


def list_images(path, file_type='images'):
    """
    列出文件夹中所有的文件，返回
    :param file_type: 'images' or 'any'
    :param path: a directory path, like '../data/pics'
    :return: all the images in the directory
    """
    IMAGE_SUFFIX = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.Png', '.PNG', '.tiff', '.bmp', '.tif']
    # IMAGE_SUFFIX = ['.png']
    paths = []
    for file_and_dir in os.listdir(path):
        if os.path.isfile(os.path.join(path, file_and_dir)):
            if file_type == 'images':
                if os.path.splitext(file_and_dir)[1] in IMAGE_SUFFIX:
                    paths.append(os.path.abspath(os.path.join(path,
                                                              file_and_dir)))
            elif file_type == 'any':
                paths.append(os.path.abspath(os.path.join(path, file_and_dir)))
            else:
                if os.path.splitext(file_and_dir)[1] == file_type:
                    paths.append(os.path.abspath(os.path.join(path,
                                                              file_and_dir)))
    return paths


def resize_image(im, max_side_len=768):
    '''
    resize image to a size multiple of 32 which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    '''
    h = im.shape[0]
    w = im.shape[1]

    resize_w = w
    resize_h = h

    # limit the max side
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = 1.
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 - 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 - 1) * 32
    im = cv2.resize(im, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im, (ratio_h, ratio_w)


class LineSegment(object):
    def __init__(self):
        # config = tf.ConfigProto(allow_soft_placement=True)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = GPU_MEMORY_FRACTION
        self.graph = tf.Graph()
        self.session = tf.Session(graph=self.graph, config=config)
        with self.session.as_default():
            with self.graph.as_default():
                if load_image_mode == cv2.IMREAD_COLOR:
                    self.input_image = tf.placeholder(tf.float32, shape=[1, None, None, 3])
                else:
                    self.input_image = tf.placeholder(tf.float32, shape=[1, None, None, 1])

                self.prediction, _ = model_builder.build_model(model_name=model_name,
                                                               frontend=frontend,
                                                               net_input=self.input_image,
                                                               num_classes=num_classes,
                                                               crop_width=IMAGE_WIDTH,
                                                               crop_height=IMAGE_HEIGHT,
                                                               dropout_p=0.0)
                saver = tf.train.Saver()
                saver.restore(self.session, ckpt_dir)
        print('Init model finished!')

    def predict_multi_classify(self, image):
        im_in = image
        im_in = np.float32(im_in) / 255.0

        if len(im_in.shape) < 3:
            im_in = np.expand_dims(im_in, axis=2)

        prediction = self.session.run(self.prediction, feed_dict={self.input_image: [im_in]})
        prediction = prediction[0]
        prediction = np.argmax(prediction, axis=-1)
        pred_colour_mask = np.zeros([im_in.shape[0], im_in.shape[1], 3], dtype=np.uint8)
        for n in range(1, num_classes):
            pos = np.vstack(np.where(prediction == n))
            pred_colour_mask[pos[0, :], pos[1, :]] = COLOUR_LIST[n]

        return prediction, pred_colour_mask

    def predict_multi_label(self, image):
        thre = 0.5
        im_in = image
        im_in = np.float32(im_in) / 255.0

        if len(im_in.shape) < 3:
            im_in = np.expand_dims(im_in, axis=2)

        prediction = self.session.run(self.prediction, feed_dict={self.input_image: [im_in]})
        prediction = sigmoid(prediction)
        prediction = prediction[0]

        score_map_list = []
        pred_mask_list = []
        for i in range(num_classes):
            pred = prediction[:, :, i]
            pred_score_map = pred * 255.
            pred_score_map = pred_score_map.astype(np.uint8)
            score_map_list.append(pred_score_map)
            # score_map_list.append(cv2.resize(pred_score_map, (image.shape[1], image.shape[0]), cv2.INTER_LINEAR))
            pred_mask = np.zeros(pred.shape, dtype=np.uint8)
            pos = np.vstack(np.where(pred > thre))
            pred_mask[pos[0, :], pos[1, :]] = 255
            pred_mask_list.append(pred_mask)
            # pred_mask_list.append(cv2.resize(pred_mask, (image.shape[1], image.shape[0]), cv2.INTER_LINEAR))

        return score_map_list, pred_mask_list


instance = LineSegment()
if __name__ == '__main__':

    delete_all_file(save_dir)

    image_name_list = list_images(root_dir)

    if class_model == 'multi_label':  # 配置多分类和多标签：multi_classify、multi_label
        for image_name in image_name_list:
            # 加载每一张待预测的图像数据
            print(image_name)
            image_name = os.path.basename(image_name)
            image_path = os.path.join(root_dir, image_name)
            image = cv2.imread(image_path, flags=load_image_mode)
            ori_height, ori_width = image.shape[0],  image.shape[1]
            # 预测
            score_map_list, pred_mask_list = instance.predict_multi_label(image)
            # 循环预测结果的每个类别map
            order_pts, label_list = [], []  # 定义标注的外框点list和类别名称list
            region_valid = False
            for n, score_map in enumerate(score_map_list):
                # inference结果二值化
                _, binary_mask = cv2.threshold(score_map_list[n], int(255. * 0.5), 255, cv2.THRESH_BINARY)
                # 闭运算
                kernel_size = 3
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
                close_mask = cv2.dilate(binary_mask, kernel)
                close_mask = cv2.erode(close_mask, kernel)
                close_mask_bak = copy.deepcopy(close_mask)
                # 连通域分析
                contours, _ = cv2.findContours(close_mask_bak, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                if len(contours) > 0:
                    contours.sort(key=lambda c: len(c), reverse=True)
                    x, y, w, h = cv2.boundingRect(contours[0])
                    if w * h > filled_area_thre[n]:
                        approx2 = cv2.approxPolyDP(contours[0], 8, True)
                        if approx2.shape[0] > 15:
                            region_valid = False
                        approx2 = approx2.reshape((approx2.shape[1], approx2.shape[0], approx2.shape[2]))
                        im = np.zeros((ori_height, ori_width), dtype="uint8")  # 获取图像的维度: (h,w)=iamge.shape[:2]
                        pts = approx2[0, :, :]
                        order_pts.append(pts)
                        if save_json:
                            label_list.append(label_name[n])

            # 存储json文件
            if save_json and (region_valid is False):
                new_order_pts = []
                for i, pts in enumerate(order_pts):
                    new_pts = []
                    for pt in pts:
                        new_pts.append([int(pt[0]), int(pt[1])])
                    new_order_pts.append(new_pts)
                generate_miniled_json(image_name, new_order_pts, label_list, 768, 768, json_path)

            debug_save_path = os.path.join(save_dir, image_name)
            if region_valid is False:
                shutil.copy(image_path, debug_save_path)

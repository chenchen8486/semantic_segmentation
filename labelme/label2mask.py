#!/usr/bin/python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
import os
import numpy as np
import cv2
import json
import time
import shutil
import codecs
from labelme.label2mask_utils import *


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


class PolygonDraw:
    def getdrawpolygonvaliddata(self, polygons):
        if isinstance(polygons, list):
            for i, polygon in enumerate(polygons):
                polygon = np.array(polygon)

                if polygon.dtype in [np.float32, np.float64]:
                    polygons[i] = np.int32(polygon)
        return polygons

    def drawfillim(self, im, polygons, fillcolor):
        try:
            im = cv2.fillPoly(im, polygons, fillcolor)  # 只使用这个函数可能会出错，不知道为啥
        except:
            try:
                im = cv2.fillConvexPoly(im, polygons, fillcolor)
            except:
                print('cant fill\n')

        return im

    @classmethod
    def drawpolygonlinesim(cls,im, polygon0, linecolor,thickness=2):
        polygon = PolygonDraw().getdrawpolygonvaliddata(polygon0)
        cv2.polylines(im, polygon, True, linecolor, thickness)
        return im

    @classmethod
    def drawpolygonfillim(cls,im, polygon0, fillcolor):
        polygon = PolygonDraw().getdrawpolygonvaliddata(polygon0)
        im=PolygonDraw().drawfillim(im, polygon, fillcolor)
        return im

    @classmethod
    def drawpolygonbinim(cls,im, polygon0, fillcolor):
        imshape = im.shape
        binim = np.zeros(imshape, np.uint8)
        polygon = PolygonDraw().getdrawpolygonvaliddata(polygon0)

        binim= PolygonDraw().drawfillim(binim, polygon, fillcolor)
        return binim


def create_label_image(image_path, json_path, generate_label_path, label_name_list, class_num):
    # 删除旧的训练数据
    delete_all_file(generate_label_path)

    # 创建ground truth需要的所有存储文件夹
    label_path, ok_path, ng_path, crop_path = create_all_need_label_file(generate_label_path)

    # 循环每张已经标注的图像数据
    bad_image = []
    img_list = list_images(image_path)
    for i, img_path in enumerate(img_list):
        if i % 100 == 0:
            print('Process %d \n' % i)
        image_base_name = os.path.basename(img_path)
        file_name = os.path.splitext(image_base_name)[0]
        json_name = file_name + '.json'

        # 为了有中文的图像名称准备，不能直接使用imread
        image = cv2.imdecode(np.fromfile((image_path + image_base_name), dtype=np.uint8), -1)

        # 有几个类别，就定义几个列表变量，每个类别分别需要创建3个列表：type_ng_mask, type_polygons, type_rects
        ng_mask_list, polygon_mask_list, rect_mask_list = [], [], []
        for label_id in range(class_num):
            ng_mask_list.append(np.zeros(image.shape, dtype=np.uint8))
            polygon_mask_list.append([])
            rect_mask_list.append([])

        # 如果在指定的json路径下，不存在与当前图像名称一样的json文件，那么就把当前图像拷贝给ok文件夹
        if not os.path.exists(os.path.join(json_path, json_name)):
            shutil.copy(img_path, ok_path)
            continue

        # 加载json文件，如果文件中"shape"这个列表为0，证明不存在标注图案，那么就把当前图像拷贝给ok文件夹
        nn = open(os.path.join(json_path, json_name), encoding='utf-8').read()
        json_data = json.loads(nn, encoding='utf-8')
        if len(json_data['shapes']) == 0:
            shutil.copy(img_path, ok_path)
            continue

        # 循环当前json中的每一个标注形状
        for cell in json_data['shapes']:
            points = cell['points'][:]
            label = cell['label']
            if cell['shape_type'] == u'polygon':
                for label_id, label_name in enumerate(label_name_list):
                    if label_name == label:
                        polygon_mask_list[label_id].append(points)
            elif cell['shape_type'] == u'rectangle':
                for label_id, label_name in enumerate(label_name_list):
                    if label_name == label:
                        rect_mask_list[label_id].append(points)
            else:
                bad_image.append(image_base_name)
                continue

        # 把找到的标注形状绘制到ng_mask上面，
        for label_id, label_name in enumerate(label_name_list):
            if len(polygon_mask_list[label_id]) > 0:
                ng_mask_list[label_id] = PolygonDraw.drawpolygonfillim(ng_mask_list[label_id],
                                                                       polygon_mask_list[label_id], 255)
            if len(rect_mask_list[label_id]) > 0:
                for points in rect_mask_list[label_id]:
                    points_flat = list()
                    for pt in points:
                        pt = map(int, pt)
                        points_flat.extend(pt)
                    np_box = np.array(points_flat)
                    np_box = np_box.reshape((len(points), 2))
                    left = np.min(np_box, axis=0)[0]
                    top = np.min(np_box, axis=0)[1]
                    right = np.max(np_box, axis=0)[0]
                    bottom = np.max(np_box, axis=0)[1]
                    cv2.rectangle(ng_mask_list[label_id], (left, top), (right, bottom), color=255, thickness=-1)

        # 把原始图像存储到ng路径下
        shutil.copy(img_path, ng_path)

        # 把ng的标注图像拷贝到labels指定的路径下
        for id in range(len(label_name_list)):
            mask_name = file_name + '_ng' + str(id) + '_mask.png'
            label_mask = ng_mask_list[id]
            if label_mask.ndim == 3:
                label_mask = cv2.cvtColor(label_mask, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(os.path.join(label_path, mask_name), label_mask)

        if len(bad_image) > 0:
            print('/////////////////////////////////')
            for cell in bad_image:
                print(cell)
        else:
            pass


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


def crop_overlap_image(input_image, direction_type, batch_w, batch_h, overlap_x=256, overlap_y=50):
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


def save_crop_image(input_image, batch_list, save_path, image_name):
    for batch_id, rect in enumerate(batch_list):
        start_x = rect[0]
        start_y = rect[1]
        end_x = rect[0] + rect[2]
        end_y = rect[1] + rect[3]
        batch_image = input_image[start_y:end_y, start_x:end_x]
        split_image_name = os.path.splitext(image_name)
        base_name = split_image_name[0]
        batch_save_path = os.path.join(save_path, base_name + "_" + str(start_x) + '_' + str(start_y) + ".bmp")
        # batch_save_path = os.path.join(save_path, str(start_x) + '_' + str(start_y) + '_' +image_name)
        cv2.imwrite(batch_save_path, batch_image)
    return 0


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


def modify_json(json_file_path, src_class_name_list, dst_class_name_list):
    '''
    描述：加载json文件，并获取json文件的数据
    :param json_path: 一个包含json文件的路径，此路径可以加载其他类型文件，不会受到影响
    :return:
    '''
    files_path_list = list_images(json_file_path, file_type='any')
    for file_id, file_path in enumerate(files_path_list):
        data_name = os.path.basename(file_path)
        extend_name = os.path.splitext(data_name)[1]
        if extend_name != '.json':  # 如果不是.json文件，跳过
            continue
        print(data_name)

        # # 加载json文件
        # if isinstance(file_path, str):
        #     file_path.encode('gb2312')
        # else:
        #     file_path.decode('utf-8').encode('gb2312')
        # json_data = json.loads(open(file_path).read(), encoding='utf-8')

        with open(file_path, 'r', encoding='utf8') as fp:
            json_data = json.load(fp)


        # 如果当前json没有做任何标注，跳过
        label_data_list = json_data['shapes']
        if len(label_data_list) == 0:
            continue
        for label_id, label_data_dict in enumerate(label_data_list[:]):
            for src_class_id, src_class in enumerate(src_class_name_list):
                if label_data_dict['label'] == src_class:
                    if dst_class_name_list[src_class_id] != '':
                        label_data_dict['label'] = dst_class_name_list[src_class_id]
                    else:
                        label_data_list.remove(label_data_dict)
        # 包存json文件(重新覆盖原本json文件即可)
        json.dump(json_data, open(file_path, 'w'), indent=1)


def calcul_label_pixel_num(label_file_path, label_name_list):
    label_num = len(label_name_list)
    ng_list = []
    each_label_white_pixel_list = []
    for label_id in range(label_num):
        ng_string = "ng" + str(label_id)
        ng_list.append(ng_string)
        each_label_white_pixel_list.append([0])


    file_list = os.listdir(label_file_path)
    for image_id, image_name in enumerate(file_list):
        image_path = os.path.join(label_file_path, image_name)

        for label_id in range(label_num):
            if ng_list[label_id] in image_name:
                label_image = cv2.imread(image_path)
                label_gray_image = cv2.cvtColor(label_image, cv2.COLOR_BGR2GRAY)
                white_pos = np.vstack(np.where(label_gray_image != 0))
                white_count = each_label_white_pixel_list[label_id][0] + white_pos[0].shape[0]
                each_label_white_pixel_list[label_id][0] = white_count

    return each_label_white_pixel_list


if __name__ == '__main__':

    # # ''' 1） 参数设置部分：'''
    # # 定义标注数据路径和生成Ground Truth的路径
    # image_path_ = 'D:/4_data/B10_cell_data/data0/total_train_data/2_label_data_with_ok/'
    # json_path_ = image_path_
    # generate_label_path_ = 'D:/4_data/B10_cell_data/data0/total_train_data/generate_gt/'
    #
    # # 定义类别数量和类别标签（在列表中有多少个类别就设置多少个字符串）
    # # label_name_list_ = ['frontier', 'mark', 'corner', 'crack', 'broken', 'chipping']
    # label_name_list_ = ['frontier', 'mark', 'corner', 'broken', 'chipping']
    # class_num_ = len(label_name_list_)
    #
    # ''' 2）生成对应的label图像：'''
    # create_label_image(image_path_, json_path_, generate_label_path_, label_name_list_, class_num_)

    # ''' 3）如果发现某个label需要合并成为一个新的label，或将某个label重新命名'''
    # json_file_path_ = 'D:/4_data'
    # # json_file_path_ = 'C:/Users'
    # src_class_name_list_ = ['crack']
    # dst_class_name_list_ = ['broken']
    # modify_json(json_file_path_, src_class_name_list_, dst_class_name_list_)


    # # ''' 4）统一对图像进行缩放处理'''
    # image_file_path =      'D:/4_data\B10_cell_data/test_http/temp2/'
    # save_train_data_path = 'D:/4_data\B10_cell_data/test_http/temp/'
    # file_list = os.listdir(image_file_path)
    # for id, image_name in enumerate(file_list):
    #     print(image_name)
    #     image_path = os.path.join(image_file_path, image_name)
    #     image_ori = cv2.imdecode(np.fromfile((image_path), dtype=np.uint8), -1)
    #     if image_ori.ndim == 3:
    #         image_gray = cv2.cvtColor(image_ori, cv2.COLOR_BGR2GRAY)
    #     else:
    #         image_gray = image_ori
    #     split_image_name = os.path.splitext(image_name)
    #     base_name = split_image_name[0]
    #     image_name = base_name + '.bmp'
    #     image_gray_resize = cv2.resize(image_gray, (0, 0), fx=1, fy=0.5, interpolation=cv2.INTER_LINEAR)
    #     cv2.imwrite(save_train_data_path + '/' + image_name, image_gray_resize)


    # # ''' 5）对训练样本进行crop，变成768/512/等等任意尺寸的小图，便于标注'''
    # # image_file_path =      'D:/4_data/B10_cell_data/data0/4_data_b10/2_ALL_resize'
    # # save_train_data_path = "D:/4_data/B10_cell_data/data0/4_data_b10/3_crop_resize_image"
    # image_file_path =      'C:/Users/admin/Desktop/test/All_resize'
    # save_train_data_path = "C:/Users/admin/Desktop/test/3_crop_resize_image"
    # crop_w, crop_h = 768, 768
    # overlap_x, overlap_y = 0, 0
    # file_list = os.listdir(image_file_path)
    # for id1, image_name in enumerate(file_list):
    #     print(image_name)
    #     #  加载原始图像数据
    #     image_path = os.path.join(image_file_path, image_name)
    #     # gray_resized_image = cv2.imread(image_path, -1)
    #     gray_resized_image = cv2.imdecode(np.fromfile((image_path), dtype=np.uint8), -1)
    #     if gray_resized_image.ndim == 3:
    #         gray_resized_image = cv2.cvtColor(gray_resized_image, cv2.COLOR_BGR2GRAY)
    #
    #     direction_type = direction_classify(gray_resized_image)
    #     crop_rect_list, crop_image_list = crop_overlap_image(gray_resized_image, direction_type, crop_w, crop_h,
    #                                                          overlap_x, overlap_y)
    #     save_crop_image(gray_resized_image, crop_rect_list, save_train_data_path, image_name)


    # # ''' 6） 对已经标注的图片和json单独剪切到另一个文件夹呢'''
    # image_file_path = "D:/4_data/B10_cell_data/data0/total_train_data/1_label_data_not_all"
    # save_train_data_path = "D:/4_data/B10_cell_data/data0/total_train_data/2_label_data_with_ok"
    # cut_same_json_and_image(image_file_path, save_train_data_path)

    # ''' 7) 统计每个类别的像素数量'''
    # label_file_path = "D:/4_data/B17_image/defect_data/5_train_data/labels"
    # label_name_list_ = ['frontier', 'mark', 'crack', 'broken', 'chipping']
    # each_label_white_pixel_list = calcul_label_pixel_num(label_file_path, label_name_list_)
    # print(each_label_white_pixel_list)
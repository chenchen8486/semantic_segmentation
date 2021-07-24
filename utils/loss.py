#!/usr/bin/env python2
# coding: utf-8
from __future__ import print_function
import os, cv2, sys, math,  time, datetime, argparse, random
import tensorflow as tf
import numpy as np

def focal_loss_weight_map(y_true_cls, y_pred_cls, n_class):
    # 生成focal loss的weight map
    gamma = 2.
    alpha = 0.5

    # 展平
    flat_y_true_cls = tf.reshape(y_true_cls, [-1, n_class])
    flat_y_pred_cls = tf.reshape(y_pred_cls, [-1, n_class])

    # 分离通道
    flat_y_true_cls_split = tf.split(flat_y_true_cls, n_class, axis=1)
    flat_y_pred_cls_split = tf.split(flat_y_pred_cls, n_class, axis=1)

    # 样本越易分，pt越接近1，则贡献的loss就越小
    # 样本越难分，pt越接近0，则贡献的loss就越小
    # pt0 = flat_y_true_cls_split[0] * flat_y_pred_cls_split[0] + \
    #       (1.0 - flat_y_true_cls_split[0]) * (1.0 - flat_y_pred_cls_split[0])
    #
    # pt1 = flat_y_true_cls_split[1] * flat_y_pred_cls_split[1] + \
    #       (1.0 - flat_y_true_cls_split[1]) * (1.0 - flat_y_pred_cls_split[1])
    #
    # pt2 = flat_y_true_cls_split[2] * flat_y_pred_cls_split[2] + \
    #       (1.0 - flat_y_true_cls_split[2]) * (1.0 - flat_y_pred_cls_split[2])

    pt_list = list()
    for n in range(n_class):
        pt = flat_y_true_cls_split[n] * flat_y_pred_cls_split[n] + \
              (1.0 - flat_y_true_cls_split[n]) * (1.0 - flat_y_pred_cls_split[n])
        pt_list.append(pt)

    # 易分的样本，pt接近1，tf.pow((1.0 - pt0), gamma)的结果就非常接近0
    # 难分的样本，pt接近0，tf.pow((1.0 - pt0), gamma)的结果就非常接近1
    # focal loss 就是通过分割的结果与gt进行比较，区分出容易分割和不容易分割的像素，然后有针对性的分配权重
    # weight_map_0 = alpha * tf.pow((1.0 - pt0), gamma)
    # weight_map_1 = alpha * tf.pow((1.0 - pt1), gamma)
    # weight_map_2 = alpha * tf.pow((1.0 - pt2), gamma)

    weight_map_list = list()
    for n in range(n_class):
        weight_map_list.append(alpha * tf.pow((1.0 - pt_list[n]), gamma))

    # 拼接通道
    if n_class == 1:
        weighted_map = weight_map_list[0]
    else:
        weighted_map = tf.concat(tuple(weight_map_list), axis=1)

    return weighted_map

# def focal_loss_weight_map(y_true_cls, y_pred_cls, n_class):
#     # 生成focal loss的weight map
#     gamma = 2.
#     alpha = 0.5
#
#     # 展平
#     flat_y_true_cls = tf.reshape(y_true_cls, [-1, n_class])
#     flat_y_pred_cls = tf.reshape(y_pred_cls, [-1, n_class])
#
#     # 分离通道
#     flat_y_true_cls_split = tf.split(flat_y_true_cls, n_class, axis=1)
#     flat_y_pred_cls_split = tf.split(flat_y_pred_cls, n_class, axis=1)
#
#     # 样本越易分，pt越接近1，则贡献的loss就越小
#     # 样本越难分，pt越接近0，则贡献的loss就越小
#     pt0 = flat_y_true_cls_split[0] * flat_y_pred_cls_split[0] + \
#         (1.0 - flat_y_true_cls_split[0]) * (1.0 - flat_y_pred_cls_split[0])
#
#     # 易分的样本，pt接近1，tf.pow((1.0 - pt0), gamma)的结果就非常接近0
#     # 难分的样本，pt接近0，tf.pow((1.0 - pt0), gamma)的结果就非常接近1
#     # focal loss 就是通过分割的结果与gt进行比较，区分出容易分割和不容易分割的像素，然后有针对性的分配权重
#     weight_map_0 = alpha * tf.pow((1.0 - pt0), gamma)
#
#     # 拼接三个通道
#     # weighted_map = tf.concat((weight_map_0, weight_map_1), axis=1)
#     weighted_map = weight_map_0
#     return weighted_map


def get_loss(logits, label_input, use_focal_loss, use_class_balance_weights, num_classes, class_model):
    # 展平
    flat_logits = tf.reshape(logits, [-1, num_classes])
    flat_labels = tf.reshape(label_input, [-1, num_classes])

    if use_focal_loss:
        # 得到focal loss计算出来的weight_map
        flat_logits = tf.nn.sigmoid(flat_logits)
        weight_map = focal_loss_weight_map(flat_labels, flat_logits, num_classes)

        # 计算交叉熵损失
        loss_map = tf.nn.sigmoid_cross_entropy_with_logits(logits=flat_logits, labels=flat_labels)

        weighted_loss = tf.multiply(loss_map, weight_map)
    else:
        # 计算交叉熵损失
        if class_model == 'multi_label':
            loss_map = tf.nn.sigmoid_cross_entropy_with_logits(logits=flat_logits, labels=flat_labels)
        elif class_model == 'multi_classify':
            loss_map = tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits, labels=flat_labels)

        weighted_loss = loss_map

    if use_focal_loss and use_class_balance_weights:
        # 使用focal loss的时候，虽然在每个类型的通道内合理的分配了容易分割和不容易分割的权重，
        # 但没有办法解决类型通道之间的前景数量不均衡的问题，如果不增加类型通道之间的相对权重的话，图例很难分割出来，原因是图例所占的比例太低，
        # 很难主导梯度的更新方向。增加类型通道权重后，图例很快就收敛了
        # 用这个方法的前提是三个类型的label是完整的，原因是，这个权重本身就是类别通道之间的权重，如果三个类型通道不完全，
        # 那意味着有些通道的loss压根就不计算，也就没必要再用上权重了。
        class_balance_weights = np.array([[1.0, 75.0, 22.15, 40.58, 58.35]])
        class_balance_weights = tf.constant(np.array(class_balance_weights, dtype=np.float32))
        # class_balance_weights = tf.multiply(flat_labels, class_balance_weights)
        # 偏置常量，目的是让背景的权重为1
        # bias = tf.constant([1], dtype=tf.float32)
        # class_balance_weights = tf.add(class_balance_weights, bias)
        weighted_loss = tf.multiply(weighted_loss, class_balance_weights)

    mean_loss = tf.reduce_mean(weighted_loss)

    return mean_loss

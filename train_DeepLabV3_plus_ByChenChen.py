#!/usr/bin/env python2
# coding: utf-8
from __future__ import print_function
import os, cv2, sys, math, time, datetime, random, argparse, json
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from utils import utils
from utils.utils import COLOUR_LIST, list_images
from utils.helpers import colour_code_segmentation, one_hot_it
from builders import model_builder
from utils.data_augmentation import *
from utils.loss import *

''' 参数配置 '''
config_path = 'D:/3_deep_leaerning_project/semantic_segmentation/config_train_tatal.json'
config_data = json.loads(open(config_path).read(), encoding='utf-8')
os.environ['CUDA_VISIBLE_DEVICES'] = config_data['CUDA_VISIBLE_DEVICES']
GPU_MEMORY_FRACTION = config_data['GPU_MEMORY_FRACTION']

IMAGE_SIZE = config_data['IMAGE_SIZE']
num_epochs = config_data['NUM_EPOCHS']
epoch_start_i = config_data['EPOCH_START']
dataset = config_data['DATASET']
crop_height = IMAGE_SIZE
crop_width = IMAGE_SIZE
h_flip = config_data['H_FLIP']
v_flip = config_data['V_FLIP']
brightness = config_data['BRIGHTNESS']
model = config_data['MODEL']
frontend = config_data['FRONTEND']

g_step = config_data['G_STEP']
num_classes = config_data['NUM_CLASSES']
CHANNEL_NUM = config_data['CHANNEL_NUM']
learning_rate = config_data['LEARNING_RATE']
BATCH_SIZE = config_data['BATCH_SIZE']
have_ok_image = config_data['HAVE_OK_IMAGE']
read_ok = config_data['READ_OK']
checkpoint_path = config_data['CHECKPOINT_PATH']
use_focal_loss = config_data['USE_FOCAL_LOSS']
use_class_balance_weights = config_data['USE_CLASS_BALANCE_WEIGHTS']
class_name = config_data['CLASS_NAME']  # 名字与mask的命名要对应,背景类不需要写，要按照优先级排列，优先级高的类型放后面
class_model = config_data['CLASS_MODEL']  # 配置多分类和多标签：multi_classify、multi_label
label_gray_diff = config_data['LABEL_GRAY_DIFF']  # 多分类的灰度差，用于tensorboard显示
image_train_dir = config_data['IMAGE_TRAIN_DIR']  # 真实样本，完整的label
mask_train_dir = config_data['MASK_TRAIN_DIR']

# ok_image_dir = config_data['OK_IMAGE_DIR']
# crop_dir = config_data['CROP_DIR']
crop_dir = None
# crop_rand_threhold = config_data['CROP_RAND_THRESHOLD']
ok_root_dir = None
# ok_root_dir = config_data['OK_ROOT_DIR']
ok_file = config_data['OK_FILE']


if __name__ == '__main__':
    # 给每个类一个colour用于生成onehot
    label_values = list()
    for i in range(num_classes):
        label_values.append(COLOUR_LIST[i])

    # 如果有ok图像，ok_iamge_list将会存储两个文件夹nothing, or， ok文件夹里面存储了所有ok图像的路径名称
    if have_ok_image:
        ok_image_list = [list_images(os.path.join(ok_root_dir, f)) for f in ok_file]

    crop_image_list = list_images(crop_dir) if (crop_dir is not None) else []

    # 输入的图像的tensor
    net_input = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNEL_NUM])

    # 输入的标注label mask的tensor
    net_output = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, num_classes])

    # 定义一个tensor变量： 全局步数
    global_step = tf.Variable(0)

    # net_input = mean_image_subtraction(net_input)  # 图像归一化

    # 建图： 构造model，调用build_model，来构造Deeplab分割模型和基础网络的Graph结构，返回logits
    logits, init_fn = model_builder.build_model(model_name=model, frontend=frontend, net_input=net_input,
                                                num_classes=num_classes, crop_width=crop_width,
                                                crop_height=crop_height, dropout_p=0.0, is_training=True)

    # 建图：  根据多标签或多分类进行损失函数计算
    if class_model == 'multi_classify':
        activation_logits = tf.nn.softmax(logits)
        loss = get_loss(logits, net_output, use_focal_loss, use_class_balance_weights, num_classes, class_model)
        gt_label = tf.argmax(net_output, axis=-1)
        gt_maps = [tf.reshape(gt_label[n, :, :], (1, IMAGE_SIZE, IMAGE_SIZE, 1)) for n in range(BATCH_SIZE)]
        pred = tf.argmax(activation_logits, axis=-1)
        # 预测结果
        pred_maps = [tf.reshape(pred[n, :, :], (1, IMAGE_SIZE, IMAGE_SIZE, 1)) for n in range(BATCH_SIZE)]
        tf.summary.image('input', net_input)
        tf.summary.scalar('loss', loss)
        for n in range(BATCH_SIZE):
            gtmap = gt_maps[n] * label_gray_diff
            gtmap = tf.to_float(gtmap, name='ToFloat')
            predmap = pred_maps[n] * label_gray_diff
            predmap = tf.to_float(predmap, name='ToFloat')
            tf.summary.image('image/score_map_%s' % str(n), gtmap)
            tf.summary.image('image/score_map_pred_%s' % str(n), predmap)
    elif class_model == 'multi_label':
        # 利用logit这个图像tensor和真实label图像tensor进行损失计算
        loss = get_loss(logits, net_output, use_focal_loss, use_class_balance_weights, num_classes, class_model)

        # 在summary显示用： 对输出的logits进行sigmoid映射
        activation_logits = tf.nn.sigmoid(logits)

        # 在tensorboard显示用： 把batch中的每个图像的所有label图像都存入gt_maps列表中
        gt_maps = list()
        for n in range(BATCH_SIZE):
            for i in range(num_classes):
                gt_maps.append(tf.reshape(net_output[n, :, :, i], (1, IMAGE_SIZE, IMAGE_SIZE, 1)))

        # 在tensorboard显示用： 把sigmoid映射后的结果都存储到pred_maps列表中
        pred_maps = list()
        for n in range(BATCH_SIZE):
            for i in range(num_classes):
                pred_maps.append(tf.reshape(activation_logits[n, :, :, i], (1, IMAGE_SIZE, IMAGE_SIZE, 1)))

        # tensorboard显示
        for n in range(BATCH_SIZE):
            tf.summary.image('input/image_%s' % str(n), tf.reshape(net_input[n, :, :, :], (1, IMAGE_SIZE, IMAGE_SIZE, 3)))
        tf.summary.scalar('loss', loss)
        for n in range(BATCH_SIZE):
            for i in range(num_classes):
                tf.summary.image('image/%s_score_map_%s' % (class_name[i], str(n)), gt_maps[n*num_classes+i])
                tf.summary.image('image/%s_score_map_pred_%s' % (class_name[i], str(n)), pred_maps[n*num_classes+i])

    # 定义优化函数
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)

    # 保存和加载模型
    saver = tf.train.Saver()
    summary_op = tf.summary.merge_all()

    # 根据NG样本路径，把所有图像路径导入列表
    train_input_names = os.listdir(image_train_dir)

    # OK样本
    # ok_image_names = os.listdir(ok_image_dir) if (ok_image_dir is not None) else None

    # 创建一个checkpoint存储文件夹，并存储ckpt
    if not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)
    summary_writer = tf.summary.FileWriter(checkpoint_path, tf.get_default_graph())

    print("\n***** Begin training *****")
    # 配置tf.Session的运算方式，比如gpu运算或者cpu运算
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # 动态控制GPU资源
    config.gpu_options.per_process_gpu_memory_fraction = GPU_MEMORY_FRACTION  # 控制GPU资源率

    # 用会话来具体执行网络, 负责分配计算资源和变量存放，以及维护执行过程中的变量
    with tf.Session(config=config) as sess:
        # 整张计算图中的变量进行初始化
        sess.run(tf.global_variables_initializer())

        # 重载历史checkpoint
        ckpt = tf.train.get_checkpoint_state(checkpoint_path)
        if ckpt:
            print('loaded ' + ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)

        # 这个函数是保证当图被多个线程共用的时候, 没有新的操作能够添加进去.
        sess.graph.finalize()

        # 对真实样本进行迭代，每迭代完一次真实样本算是一次epoch
        for epoch in range(epoch_start_i, num_epochs):
            # 打乱样本顺序（返回一个列表，列表中存储着最大尺寸内的随机数字）
            id_list = np.random.permutation(len(train_input_names))

            # 本轮epoch需要迭代的次数==真实样本的数量
            num_iters = int(np.floor(len(id_list)))

            # 训练策略是NG样本训练5次后训练1次OK样本
            i_counter = 0
            b = 0
            neg_flag = False
            input_image_batch = []
            output_image_batch = []
            while i_counter < num_iters:
                i = i_counter
                id = id_list[i]

                # 读取image 和 label
                score_map_in = []

                if i != 0 and i % read_ok == 0 and (neg_flag is False) and have_ok_image:
                    # OK
                    neg_flag = True

                    image_path = random.choice(ok_image_list[random.choice([x for x in range(len(ok_file))])])

                    if CHANNEL_NUM == 3:
                        # im_in = cv2.imread(image_path)
                        im_in = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                    else:
                        #im_in = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                        im_in = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

                    for n in range(num_classes):
                        score_map_in.append(np.zeros([im_in.shape[0], im_in.shape[1]], dtype=np.uint8))

                    if num_classes == 1:
                        im_in, label = random_paste_croped_image(crop_image_list, im_in, score_map_in[0])
                        score_map_in = [label]

                else:
                    # NG
                    i_counter += 1
                    neg_flag = False
                    base_name = train_input_names[id]
                    image_path = os.path.join(image_train_dir, base_name)
                    # 加载： 加载原始的训练图像（单张）
                    if CHANNEL_NUM == 3:
                        #im_in = cv2.imread(image_path)
                        im_in = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                    else:
                        #im_in = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                        im_in = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                    # 多标签
                    if class_model == 'multi_label':
                        # 加载： 加载每一个类别的label mask， 存储到score_map_in列表中 （多张）
                        for n in range(num_classes):
                            if '.bmp' in base_name:
                                mask_path = os.path.join(mask_train_dir, base_name.replace('.bmp', '_%s_mask.png' % class_name[n]))
                            elif '.jpg' in base_name:
                                mask_path = os.path.join(mask_train_dir, base_name.replace('.jpg', '_%s_mask.png' % class_name[n]))
                            else:
                                mask_path = os.path.join(mask_train_dir, base_name.replace('.png', '_%s_mask.png' % class_name[n]))
                            #score_map_in.append(cv2.imread(mask_path, flags=0))
                            score_map_in.append(cv2.imdecode(np.fromfile(mask_path, dtype=np.uint8), 0))

                        if num_classes == 1:
                            im_in, label = random_paste_croped_image(crop_image_list, im_in, score_map_in[0])
                            score_map_in = [label]
                    # 多分类
                    elif class_model == 'multi_classify':

                        # 先给一张背景类
                        score_map_in.append(np.zeros([im_in.shape[0], im_in.shape[1]], dtype=np.uint8))

                        # 再加载缺陷类的mask,循环时需要减去背景类
                        for n in range(num_classes-1):
                            if '.bmp' in base_name:
                                mask_path = os.path.join(mask_train_dir, base_name.replace('.bmp', '_%s_mask.png' % class_name[n]))
                            else:
                                mask_path = os.path.join(mask_train_dir, base_name.replace('.png', '_%s_mask.png' % class_name[n]))
                            #score_map_in.append(cv2.imread(mask_path, flags=0))
                            score_map_in.append(cv2.imdecode(np.fromfile(mask_path, dtype=np.uint8), 0))

                # 数据增强：随机旋转和翻转
                im_in, score_map_in = random_rotate_and_flip(im_in, score_map_in)
                im_in, score_map_in = random_stretch(im_in, score_map_in)

                # 数据增强后，需要限制图像的最大边长，原图的双线性。label用双线性，最近邻当前景太细太小，会有像素上的损失
                im_in, _ = resize_image(im_in, interpolation='INTER_LINEAR')
                score_map_in = [resize_image(cell, interpolation='INTER_LINEAR')[0] for cell in score_map_in]

                # 数据增强： 随机缩放和拉伸
                im_in, score_map_in = random_resize_and_paste(im_in, score_map_in)

                # 数据增强： 随机亮度和质量
                im_in = random_jpg_quality(im_in)
                im_in = random_bright(im_in)

                # 单张训练图像和n个类别的label mask 转换成浮点
                im_in = im_in.astype(np.float32)
                score_map_in = [cell.astype(np.float32) for cell in score_map_in]

                # 如果生成的掩模图像是0-255的，在这里就被转换为0-1的图像了
                for cell in score_map_in:
                    cell[cell > 0.] = 1.

                # 把n类的label mask进行合并，列表->np.stack = output_image
                if len(score_map_in) == 1:
                    # 单分类
                    output_image = score_map_in[0]
                else:
                    if class_model == 'multi_label':
                        output_image = np.stack(np.array(score_map_in), axis=2)
                    elif class_model == 'multi_classify':
                        gt = np.zeros([im_in.shape[0], im_in.shape[1], num_classes], dtype=np.uint8)
                        for n in range(1, len(score_map_in)):
                            pos = np.vstack(np.where(score_map_in[n] > 0.))
                            gt[pos[0, :], pos[1, :]] = label_values[n]
                            semantic_map = one_hot_it(gt, label_values)
                            output_image = semantic_map.astype(np.float)

                # 训练图像存入列表： 将输入图像放进batch的列表中
                input_image = np.float32(im_in) / 255.
                input_image_batch.append(input_image)
                # 训练label mask存入列表： 将标注好的label mask的np.stack存入到列表中（一个列表元素等价一个stack，一个batch有m个stack）
                output_image_batch.append(output_image)

                # 如果batch列表中已经装满数据就进入循环
                if len(input_image_batch) == BATCH_SIZE:
                    g_step += 1
                    # 已经把batch的数量改成只能为1
                    input_image_batch = np.array(input_image_batch)
                    output_image_batch = np.array(output_image_batch)

                    if len(output_image_batch.shape) < 4:
                        output_image_batch = np.expand_dims(output_image_batch, axis=4)

                    if len(input_image_batch.shape) < 4:
                        input_image_batch = np.expand_dims(input_image_batch, axis=4)

                    # Do the training
                    _, current, summary_str = sess.run([opt, loss, summary_op], feed_dict={net_input: input_image_batch,
                                                                                           net_output: output_image_batch})
                    if g_step % 50 == 0:
                        string_print = "g_step = %d Current_Loss = %.6f " % (g_step, current)
                        utils.LOG(string_print)

                    if g_step % 50 == 0:
                        saver.save(sess, checkpoint_path + "/model.ckpt", global_step=global_step)

                    if g_step % 50 == 0:
                        # g_step % 50 == 1 的时候存真实样本
                        # g_step % 50 == 0 的时候存附加样本
                        summary_writer.add_summary(summary_str, global_step=g_step)
                    input_image_batch = []
                    output_image_batch = []
                else:
                    continue

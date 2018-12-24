#!/usr/bin/env python
# encoding: utf-8
'''
@Time    : 2018/12/22 16:55
@Author  : Xu FJ
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import json
import logging
import os
import sys
import random
from random import shuffle

import numpy as np

import scipy as scp
import scipy.misc
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.training import queue_runner
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.framework import dtypes

import threading


def create_queues(hypes, phase):
    """Create Queues."""
    arch = hypes['arch']
    dtypes = [tf.float32, tf.int32]

    shape_known = hypes['jitter']['reseize_image'] \
                  or hypes['jitter']['crop_patch']

    if shape_known:
        if hypes['jitter']['crop_patch']:
            height = hypes['jitter']['patch_height']
            width = hypes['jitter']['patch_width']
        else:
            height = hypes['jitter']['image_height']
            width = hypes['jitter']['image_width']
        channel = hypes['arch']['num_channels']
        num_classes = hypes['arch']['num_classes']
        shapes = [[height, width, channel],
                  [height, width, num_classes]]
    else:
        shapes = None

    capacity = 50
    q = tf.FIFOQueue(capacity=50, dtypes=dtypes, shapes=shapes)
    tf.summary.scalar("queue/%s/fraction_of_%d_full" %
                      (q.name + "_" + phase, capacity),
                      math_ops.cast(q.size(), tf.float32) * (1. / capacity))

    return q


def _make_data_gen(hypes, phase, data_dir):
    """Return a data generator that outputs image samples.

    @ Returns
    image: integer array of shape [height, width, 3].
    Representing RGB value of each pixel.
    gt_image: boolean array of shape [height, width, num_classes].
    Set `gt_image[i,j,k] == 1` if and only if pixel i,j
    is assigned class k. `gt_image[i,j,k] == 0` otherwise.

    [Alternativly make gt_image[i,j,*] a valid propability
    distribution.]
    """
    if phase == 'train':
        data_file = hypes['data']["train_file"]
    elif phase == 'val':
        data_file = hypes['data']["val_file"]
    else:
        assert False, "Unknown Phase %s" % phase

    data_file = os.path.join(data_dir, data_file)

    road_color = np.array(hypes['data']['road_color'])
    background_color = np.array(hypes['data']['background_color'])

    data = _load_gt_file(hypes, data_file)

    # 制作gt
    for image, gt_image in data:
        # 转bool
        gt_bg = np.all(gt_image == background_color, axis=2)
        gt_road = np.all(gt_image == road_color, axis=2)

        # reshape标签，添加1通道
        assert (gt_road.shape == gt_bg.shape)
        shape = gt_bg.shape
        gt_bg = gt_bg.reshape(shape[0], shape[1], 1)
        gt_road = gt_road.reshape(shape[0], shape[1], 1)

        # 第一个变量为背景，第二个为目标（道路）
        gt_image = np.concatenate((gt_bg, gt_road), axis=2)

        if phase == 'val':
            yield image, gt_image
        elif phase == 'train':

            yield jitter_input(hypes, image, gt_image)

            yield jitter_input(hypes, np.fliplr(image), np.fliplr(gt_image))


def _load_gt_file(hypes, data_file=None):
    """Take the data_file and hypes and create a generator.

    The generator outputs the image and the gt_image.
    """
    base_path = os.path.realpath(os.path.dirname(data_file))
    files = [line.rstrip() for line in open(data_file)]

    for epoche in itertools.count():
        shuffle(files)
        for file in files:
            image_file, gt_image_file = file.split(" ")
            image_file = os.path.join(base_path, image_file)
            assert os.path.exists(image_file), \
                "File does not exist: %s" % image_file
            gt_image_file = os.path.join(base_path, gt_image_file)
            assert os.path.exists(gt_image_file), \
                "File does not exist: %s" % gt_image_file
            # 1242*375
            image = scipy.misc.imread(image_file, mode='RGB')
            # Please update Scipy, if mode='RGB' is not avaible
            gt_image = scp.misc.imread(gt_image_file, mode='RGB')

            yield image, gt_image


def jitter_input(hypes, image, gt_image):
    jitter = hypes['jitter']
    res_chance = jitter['res_chance']
    crop_chance = jitter['crop_chance']

    if jitter['random_resize'] and res_chance > random.random():
        lower_size = jitter['lower_size']
        upper_size = jitter['upper_size']
        sig = jitter['sig']
        image, gt_image = random_resize(image, gt_image, lower_size,
                                        upper_size, sig)
        image, gt_image = crop_to_size(hypes, image, gt_image)

    if jitter['random_crop'] and crop_chance > random.random():
        max_crop = jitter['max_crop']
        crop_chance = jitter['crop_chance']
        image, gt_image = random_crop_soft(image, gt_image, max_crop)

    if jitter['reseize_image']:
        image_height = jitter['image_height']
        image_width = jitter['image_width']
        image, gt_image = resize_label_image(image, gt_image,
                                             image_height,
                                             image_width)

    if jitter['crop_patch']:
        patch_height = jitter['patch_height']
        patch_width = jitter['patch_width']
        image, gt_image = random_crop(image, gt_image,
                                      patch_height, patch_width)

    assert (image.shape[:-1] == gt_image.shape[:-1])
    return image, gt_image


def crop_to_size(hypes, image, gt_image):
    new_width = image.shape[1]
    new_height = image.shape[0]
    width = hypes['arch']['image_width']
    height = hypes['arch']['image_height']
    if new_width > width:
        max_x = max(new_height - height, 0)
        max_y = new_width - width
        offset_x = random.randint(0, max_x)
        offset_y = random.randint(0, max_y)
        image = image[offset_x:offset_x + height, offset_y:offset_y + width]
        gt_image = gt_image[offset_x:offset_x + height, offset_y:offset_y + width]

    return image, gt_image


def random_crop(image, gt_image, height, width):
    old_width = image.shape[1]
    old_height = image.shape[0]
    assert (old_width >= width)
    assert (old_height >= height)
    max_x = max(old_height - height, 0)
    max_y = max(old_width - width, 0)
    offset_x = random.randint(0, max_x)
    offset_y = random.randint(0, max_y)
    image = image[offset_x:offset_x + height, offset_y:offset_y + width]
    gt_image = gt_image[offset_x:offset_x + height, offset_y:offset_y + width]

    assert (image.shape[0] == height)
    assert (image.shape[1] == width)

    return image, gt_image


def random_crop_soft(image, gt_image, max_crop):
    offset_x = random.randint(1, max_crop)
    offset_y = random.randint(1, max_crop)

    if random.random() > 0.5:
        image = image[offset_x:, offset_y:, :]
        gt_image = gt_image[offset_x:, offset_y:, :]
    else:
        image = image[:-offset_x, :-offset_y, :]
        gt_image = gt_image[:-offset_x, :-offset_y, :]

    return image, gt_image


def resize_label_image_with_pad(image, label, image_height, image_width):
    shape = image.shape
    assert (image_height >= shape[0])
    assert (image_width >= shape[1])

    pad_height = image_height - shape[0]
    pad_width = image_width - shape[1]
    offset_x = random.randint(0, pad_height)
    offset_y = random.randint(0, pad_width)

    new_image = np.zeros([image_height, image_width, 3])
    new_image[offset_x:offset_x + shape[0], offset_y:offset_y + shape[1]] = image

    new_label = np.zeros([image_height, image_width, 2])
    new_label[offset_x:offset_x + shape[0], offset_y:offset_y + shape[1]] = label

    return new_image, new_label


def resize_label_image(image, gt_image, image_height, image_width):
    image = scipy.misc.imresize(image, size=(image_height, image_width),
                                interp='cubic')
    shape = gt_image.shape
    gt_zero = np.zeros([shape[0], shape[1], 1])
    gt_image = np.concatenate((gt_image, gt_zero), axis=2)
    gt_image = scipy.misc.imresize(gt_image, size=(image_height, image_width),
                                   interp='nearest')
    gt_image = gt_image[:, :, 0:2] / 255

    return image, gt_image


def random_resize(image, gt_image, lower_size, upper_size, sig):
    factor = random.normalvariate(1, sig)
    if factor < lower_size:
        factor = lower_size
    if factor > upper_size:
        factor = upper_size
    image = scipy.misc.imresize(image, factor)
    shape = gt_image.shape
    gt_zero = np.zeros([shape[0], shape[1], 1])
    gt_image = np.concatenate((gt_image, gt_zero), axis=2)
    gt_image = scipy.misc.imresize(gt_image, factor, interp='nearest')
    gt_image = gt_image[:, :, 0:2] / 255
    return image, gt_image


def crop_to_size(hypes, image, gt_image):
    new_width = image.shape[1]
    new_height = image.shape[0]
    width = hypes['arch']['image_width']
    height = hypes['arch']['image_height']
    if new_width > width:
        max_x = max(new_height - height, 0)
        max_y = new_width - width
        offset_x = random.randint(0, max_x)
        offset_y = random.randint(0, max_y)
        image = image[offset_x:offset_x + height, offset_y:offset_y + width]
        gt_image = gt_image[offset_x:offset_x + height, offset_y:offset_y + width]

    return image, gt_image


def inputs(hypes, q, phase):
    """Generate Inputs images."""
    if phase == 'val':
        image, label = q.dequeue()
        image = tf.expand_dims(image, 0)
        label = tf.expand_dims(label, 0)
        return image, label

    shape_known = hypes['jitter']['reseize_image'] or hypes['jitter']['crop_patch']

    if not shape_known:
        image, label = q.dequeue()
        # nc为类别数，背景与道路
        nc = hypes["arch"]["num_classes"]
        label.set_shape([None, None, nc])
        image.set_shape([None, None, 3])
        # (none,none,3)->(1,none,none,3)增加1维batch_size
        image = tf.expand_dims(image, 0)
        label = tf.expand_dims(label, 0)
        if hypes['solver']['batch_size'] > 1:
            logging.error("Using a batch_size of {} with unknown shape."
                          .format(hypes['solver']['batch_size']))
            logging.error("Set batch_size to 1 or use `reseize_image` "
                          "or `crop_patch` to obtain a defined shape")
            raise ValueError
    else:
        image, label = q.dequeue_many(hypes['solver']['batch_size'])

    image = _processe_image(hypes, image)

    # Display the training images in the visualizer.
    tensor_name = image.op.name
    tf.summary.image(tensor_name + '/image', image)

    road = tf.expand_dims(tf.to_float(label[:, :, :, 0]), 3)
    tf.summary.image(tensor_name + '/gt_image', road)

    return image, label


def _processe_image(hypes, image):
    # 不改变形状
    # Because these operations are not commutative, consider randomizing
    # randomize the order their operation.
    augment_level = hypes['jitter']['augment_level']
    if augment_level > 0:
        image = tf.image.random_brightness(image, max_delta=30)
        image = tf.image.random_contrast(image, lower=0.75, upper=1.25)
    if augment_level > 1:
        image = tf.image.random_hue(image, max_delta=0.15)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.6)

    return image


def start_enqueuing_threads(hypes, q, phase, sess):
    """Start enqueuing threads."""
    image_pl = tf.placeholder(tf.float32)
    label_pl = tf.placeholder(tf.int32)
    data_dir = hypes['dirs']['data_dir']

    def make_feed(data):
        image, label = data
        return {image_pl: image, label_pl: label}

    def enqueue_loop(sess, enqueue_op, phase, gen):
        # infinity loop enqueueing data
        for d in gen:
            sess.run(enqueue_op, feed_dict=make_feed(d))

    enqueue_op = q.enqueue((image_pl, label_pl))
    gen = _make_data_gen(hypes, phase, data_dir)
    gen.__next__()
    # sess.run(enqueue_op, feed_dict=make_feed(data))
    if phase == 'val':
        num_threads = 1
    else:
        num_threads = 1
    for i in range(num_threads):
        t = threading.Thread(target=enqueue_loop,
                             args=(sess, enqueue_op,
                                   phase, gen))
        t.daemon = True
        t.start()


def main():
    """main."""
    with open('../config/fcn8_seg.json', 'r') as f:
        hypes = json.load(f)

    q = {}
    q['train'] = create_queues(hypes, 'train')
    q['val'] = create_queues(hypes, 'val')
    data_dir = "../data"
    _make_data_gen(hypes, 'train', data_dir)
    image_batch, label_batch = inputs(hypes, q['train'], 'train')

    logging.info("Start running")

    with tf.Session() as sess:
        # Run the Op to initialize the variables.
        init = tf.initialize_all_variables()
        sess.run(init)
        coord = tf.train.Coordinator()
        start_enqueuing_threads(hypes, q['train'], 'train', sess)

        logging.info("Start running")
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for i in itertools.count():
            image = image_batch.eval()
            gt = label_batch.eval()
            plt.imshow(np.uint8(image[0]))
            plt.show()
            gt_bg = gt[0, :, :, 0]
            gt_road = gt[0, :, :, 1]
            plt.imshow(np.uint8(gt_bg))
            plt.show()
            plt.imshow(np.uint8(gt_road))
            plt.show()
            print(1)

        coord.request_stop()
        coord.join(threads)


# exmaple:
# with tf.Session() as sess:
#     # 先执行初始化工作
#     sess.run(tf.global_variables_initializer())
#     sess.run(tf.local_variables_initializer())
#
#     # 开启一个协调器
#     coord = tf.train.Coordinator()
#     # 使用start_queue_runners 启动队列填充
#     threads = tf.train.start_queue_runners(sess, coord)
#
#     try:
#         while not coord.should_stop():
#             print
#             '************'
#             # 获取每一个batch中batch_size个样本和标签
#             image_batch_v, label_batch_v = sess.run([image_batch, label_batch])
#             print(image_batch_v.shape, label_batch_v)
#     except tf.errors.OutOfRangeError:  # 如果读取到文件队列末尾会抛出此异常
#         print("done! now lets kill all the threads……")
#     finally:
#         # 协调器coord发出所有线程终止信号
#         coord.request_stop()
#         print('all threads are asked to stop!')
#     coord.join(threads)  # 把开启的线程加入主线程，等待threads结束
#     print('all threads are stopped!')
# https://s3.eu-central-1.amazonaws.com/avg-kitti/data_road.zip

if __name__ == '__main__':
    main()

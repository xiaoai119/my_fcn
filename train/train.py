#!/usr/bin/env python
# encoding: utf-8
'''
@Time    : 2019/1/2 19:09
@Author  : Xu FJ
'''
import logging

import commentjson
import tensorflow as tf

from decoder import multiloss
from encoder import fcn8_vgg
from inputs import fcn_seg_input
from optimizer import generic_optimizer
import tensorvision.train as train

def main():
    with open('../config/fcn8_seg.json', 'r') as f:
        logging.info("f: %s", f)
        hypes = commentjson.load(f)

    # utils.set_dirs(hypes, '../config/fcn8_seg.json')
    # utils._add_paths_to_sys(hypes)
    # logging.info("Initialize training folder")
    # train.initialize_training_folder(hypes)
    # logging.info("Start training")
    # train.do_training(hypes)
    learning_rate = tf.placeholder(tf.float32)

    with tf.name_scope("Queues"):
        queue = fcn_seg_input.create_queues(hypes, 'train')

    with tf.name_scope("Inputs"):
        image, labels = fcn_seg_input.inputs(hypes, queue, phase='train')

    # Run inference on the encoder network
    logits = fcn8_vgg.inference(hypes, image, train=True)

    # Build decoder on top of the logits
    decoded_logits = multiloss.decoder(hypes, logits, train=True)

    # Add to the Graph the Ops for loss calculation.
    with tf.name_scope("Loss"):
        losses = multiloss.loss(hypes, decoded_logits,
                                labels)

    # Add to the Graph the Ops that calculate and apply gradients.
    with tf.name_scope("Optimizer"):
        global_step = tf.Variable(0, trainable=False)
        # Build training operation
        train_op = generic_optimizer.training(hypes, losses,
                                              global_step, learning_rate)

        # with tf.name_scope("Evaluation"):
        #     # Add the Op to compare the logits to the labels during evaluation.
        #     eval_list = eval.(
        #         hypes, image, labels, decoded_logits, losses, global_step)

        summary_op = tf.summary.merge_all()
    # with tf.Session() as sess:
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 1
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

        # sess = tf.get_default_session()
        init = tf.global_variables_initializer()
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        fcn_seg_input.start_enqueuing_threads(hypes, queue, 'train', sess)

        for step in range(0, 1):
            lr = 0.0001
            feed_dict = {learning_rate: lr}
            run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)
            sess.run(train_op, feed_dict=feed_dict, options=run_options)
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    main()

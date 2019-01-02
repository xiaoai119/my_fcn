#!/usr/bin/env python
# encoding: utf-8
'''
@Time    : 2019/1/2 19:09
@Author  : Xu FJ
'''
import tensorflow as tf

import commentjson
import logging
import os
import sys

import collections
import tensorvision.train as train
import tensorvision.utils as utils


def main():
    utils.set_gpus_to_use()

    with open('../config/fcn8_seg.json', 'r') as f:
        logging.info("f: %s", f)
        hypes = commentjson.load(f)

    logging.info("Initialize training folder")
    train.initialize_training_folder(hypes)
    logging.info("Start training")
    train.do_training(hypes)


if __name__ == '__main__':
    main()

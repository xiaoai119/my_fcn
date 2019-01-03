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

    with open('../config/fcn8_seg.json', 'r') as f:
        logging.info("f: %s", f)
        hypes = commentjson.load(f)

    utils.set_dirs(hypes, '../config/fcn8_seg.json')
    utils._add_paths_to_sys(hypes)
    logging.info("Initialize training folder")
    train.initialize_training_folder(hypes)
    logging.info("Start training")
    train.do_training(hypes)


if __name__ == '__main__':
    main()

# !/usr/bin/env python
# -*- encoding: utf-8 -*-
# Version:      python 3.6
# Description:  
# Author:       bqrmtao@qq.com
# date:         2021/10/21 11:45

from __future__ import absolute_import

import os
import sys
sys.path.append(os.getcwd())

from modeling.detection.yolo_v3 import YoloV3

from utils.logger import logger

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset_path",
        type=str,
        default="./dataset/",
    )
    parser.add_argument(
        "batch_size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "num_epoch",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--filter_classes",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--model_def",
        type=str,
        default="./modeling/backbone/darknet/yolov3_tiny.cfg",
    )
    parser.add_argument(
        "--pretrained_weights",
        type=str,
        default="./init_weights/darknet/darknet53.conv.74",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--conf_thresh",
        type=float,
        default=0.4,
    )
    parser.add_argument(
        "--nms_thresh",
        type=float,
        default=0.4,
    )
    parser.add_argument(
        "--gradient_accumulation",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--ckpt_interval",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="./checkpoint/detection/yolo_v3/",
    )
    parser.add_argument(
        "--result_path",
        type=str,
        default="./result/detection/yolo_v3/",
    )

    args = parser.parse_args()
    logger.debug(args)

    model = YoloV3(
        model_def=args.model_def, pretrained_weights=args.pretrained_weights, filter_classes=args.filter_classes,
        ckpt_path=args.ckpt_path, result_path=args.result_path,
    )

    # logger.info("train with nms_thresh:{} conf_thresh:{}".format(kwargs["nms_thresh"], kwargs["conf_thresh"]))
    model.train(
        args.dataset_path, args.num_epoch, args.batch_size,
        # nms_thresh=nms_thresh / 100, conf_thresh=conf_thresh / 100,
        ckpt_interval=args.ckpt_interval,
        gradient_accumulation=args.gradient_accumulation,
    )

    logger.info("{} epoch(s) trained".format(args.num_epoch))

    # logger.info("begin to evaluate")
    # model.evaluate()
    logger.info("all done")


if "__main__" == __name__:
    main()

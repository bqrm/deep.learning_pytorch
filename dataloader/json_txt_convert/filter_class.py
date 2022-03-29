# !/usr/bin/env python
# -*- encoding: utf-8 -*-
# Version:		python 3.6.5
# Description:	
# Author:		bqrmtao@qq.com
# date:			2021/3/31 17:21

from __future__ import absolute_import

import os
import sys
sys.path.append(os.getcwd())

import base64
import json
import numpy as np
import shutil

from utils.common_method import get_filename_iteratively


dst_image_path = r"./dataset/coco_car_cat/val/image"
dst_annotation_path = r"./dataset/coco_car_cat/val/annotation"
os.makedirs(dst_image_path, exist_ok=True)
os.makedirs(dst_annotation_path, exist_ok=True)

src_image_path = r"./dataset/coco_1/val/image"
src_annotation_path = r"./dataset/coco_1/val/annotation"

list_image_filename = []
list_annotation_filename = []
get_filename_iteratively(src_image_path, list_image_filename)
get_filename_iteratively(src_annotation_path, list_annotation_filename)


for annotation_file_idx, annotation_file_path in enumerate(list_annotation_filename):
    # print("{}: {}".format(annotation_file_idx, annotation_file_path))
    if annotation_file_idx % 1000 == 0:
        print("{}: {}".format(annotation_file_idx, annotation_file_path))

    basename, _ = os.path.splitext(os.path.basename(annotation_file_path))
    image_file_path = ""
    for tmp_file_path in list_image_filename:
        if basename in tmp_file_path:
            image_file_path = tmp_file_path
            break

    if "" == image_file_path:
        continue

    with open(annotation_file_path, 'r') as f:
        coco_data = json.load(f)

    info_idx = 0
    while len(coco_data["shapes"]) > info_idx:
        if coco_data["shapes"][info_idx]["label"] not in ["cat", "car"]:
            coco_data["shapes"].remove(coco_data["shapes"][info_idx])
        else:
            info_idx += 1

    if info_idx == 0:
        continue

    target_name = image_file_path.replace(src_image_path, dst_image_path)
    shutil.copy(image_file_path, target_name)

    target_name = annotation_file_path.replace(src_annotation_path, dst_annotation_path)
    with open(target_name, "wb") as f:
        f.write(json.dumps(coco_data, indent=2).encode())






























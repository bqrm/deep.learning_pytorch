# !/usr/bin/env python
# -*- encoding: utf-8 -*-
# Version:		python 3.6.5
# Description:	
# Author:		bqrmtao@qq.com
# date:			2021/1/13 14:05

from __future__ import absolute_import

import os
import sys
sys.path.append(os.getcwd())

import base64
import json
import numpy as np

from collections import defaultdict

from utils.logger import logger

dst_path = r"D:/Project/DataSet/coco/val/annotation/"
os.makedirs(dst_path, exist_ok=True)

src_filename = r"D:/Project/DataSet/coco/annotation/instances_val2014.json"
with open(src_filename, 'r') as f:
    coco_data = json.load(f)
logger.info("{} image(s), {} annotation(s), {} categories loaded".format(len(coco_data["images"]), len(coco_data["annotations"]), len(coco_data["categories"])))

image_path = r"D:/Project/DataSet/coco/val/image/"

dict_category = {}
for category_info in coco_data["categories"]:
    dict_category[category_info["id"]] = category_info["name"]

image_id_to_idx = {}
for image_idx in range(len(coco_data["images"])):
    # image_info = coco_data["images"][image_idx]
    image_name = int(coco_data["images"][image_idx]["file_name"][15:-4])
    image_id_to_idx[image_name] = image_idx

annotation_dict = defaultdict(list)
for annotation_info in coco_data["annotations"]:
    image_idx = annotation_info["image_id"]

    if image_idx in image_id_to_idx:
        annotation_dict[image_idx].append(annotation_info)
    else:
        logger.info("no annotation for image {}".format(coco_data["images"][image_idx]["file_name"]))

logger.info("annotation of {} image(s) parsed".format(len(annotation_dict)))

for annotation_idx, annotation_key in enumerate(annotation_dict):
    if annotation_idx % 1000 == 0:
        print("{}".format(annotation_idx))
    
    annotation_info = annotation_dict[annotation_key]

    image_info = coco_data["images"][image_id_to_idx[annotation_key]]

    image_name = os.path.join(image_path, image_info["file_name"])
    if os.path.exists(image_name):
        with open(image_name, "rb") as f:
            str_image_base64 = base64.b64encode(f.read()).decode("ascii")
    else:
        logger.warn("image {} not exist".format(image_name))

    json_data = {
        "version": "4.5.6",
        "flags": {},
        "shapes": [],
        "imagePath": image_info["file_name"],
        "imageData": None,
        # "imageData": str_image_base64,
        "imageHeight": image_info["height"],
        "imageWidth": image_info["width"],
        "imageID": annotation_key,
    }

    for bounding_box_info in annotation_info:
        shape_info = {
            "label": dict_category[bounding_box_info["category_id"]],
            "points": [
                [
                    int(round(bounding_box_info["bbox"][0])),
                    int(round(bounding_box_info["bbox"][1])),
                ],
                [
                    int(round(bounding_box_info["bbox"][0] + bounding_box_info["bbox"][2])),
                    int(round(bounding_box_info["bbox"][1] + bounding_box_info["bbox"][3])),
                ],
            ],
            "shape_type": "rectangle",
            "flags": {}
        }
        json_data["shapes"].append(shape_info)
    
    base_name, _ = os.path.splitext(image_info["file_name"])
    target_name = os.path.join(dst_path, base_name + ".json")
    with open(target_name, "wb") as f:
        f.write(json.dumps(json_data, indent=2).encode())
































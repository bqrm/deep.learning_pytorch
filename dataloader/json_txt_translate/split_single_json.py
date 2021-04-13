# !/usr/bin/env python
# -*- encoding: utf-8 -*-
# Version:		python 3.6.5
# Description:	
# Author:		bqrmtao@qq.com
# date:			2021/1/13 14:05

from __future__ import absolute_import

import base64
import json
import numpy as np
import os

from collections import defaultdict

dst_path = r"/home/taomingyang/dataset/coco/annotations/val2014"
os.makedirs(dst_path, exist_ok=True)

src_filename = r"/home/taomingyang/dataset/coco/annotations/instances_val2014.json"
with open(src_filename, 'r') as f:
    coco_data = json.load(f)

image_path = r"/home/taomingyang/dataset/coco/images/val2014"

categories = {}
for cate_info in coco_data["categories"]:
    categories[cate_info["id"]] = cate_info["name"]

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

for annotation_key in annotation_dict:
    annotation_info = annotation_dict[annotation_key]

    image_info = coco_data["images"][image_id_to_idx[annotation_key]]

    image_name = os.path.join(image_path, image_info["file_name"])
    with open(image_name, "rb") as f:
        str_image_base64 = base64.b64encode(f.read()).decode("ascii")

    json_data = {
        "version": "4.5.6",
        "flags": {},
        "shapes": [],
        "imagePath": image_info["file_name"],
        "imageData": None,
        # "imageData": str_image_base64,
        "imageHeight": image_info["height"],
        "imageWidth": image_info["width"],
    }

    for bounding_box_info in annotation_info:
        shape_info = {
            "label": categories[bounding_box_info["category_id"]],
            "points": [
                bounding_box_info["bbox"][0:2],
                [bounding_box_info["bbox"][0] + bounding_box_info["bbox"][2], bounding_box_info["bbox"][1] + bounding_box_info["bbox"][3]],
            ],
            "shape_type": "rectangle",
            "flags": {}
        }
        json_data["shapes"].append(shape_info)
    
    base_name, _ = os.path.splitext(image_info["file_name"])
    target_name = os.path.join(dst_path, base_name + ".json")
    with open(target_name, "wb") as f:
        f.write(json.dumps(json_data, indent=2).encode())
































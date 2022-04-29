# -*- coding: utf-8 -*-
# Version:		python 3.6.5
# Description:
# Author:		bqrmtao@qq.com
# date:			2019/7/10

import os
import sys
sys.path.append(os.getcwd())

import json

from utils.common_method import get_filename_iteratively


annotation_path = r"D:/Project/DataSet/coco_car_and_cat/train/annotation/"
label_path = r"D:/Project/DataSet/coco_car_and_cat/train/label/"
image_path_txt = r"D:/Project/DataSet/coco_car_and_cat/train/file_path.txt"
class_path = r"D:/Project/DataSet/coco_car_and_cat/classes.names"

os.makedirs(label_path, exist_ok=True)

list_filename = []
get_filename_iteratively(annotation_path, list_filename)

dict_label = dict()
class_handler = open(class_path, 'r', encoding="utf-8")
class_line = class_handler.readlines()
class_handler.close()

for class_idx, class_name in enumerate(class_line):
    dict_label[class_name[:-1]] = class_idx

image_path_handler = open(image_path_txt, "w", encoding="utf-8")

for file_idx, annotation_filename in enumerate(list_filename):
    if file_idx % 1000 == 0:
        print("%d: %s" % (file_idx, annotation_filename))

    read_handler = open(annotation_filename, "rb")
    location_info = json.load(read_handler)
    read_handler.close()

    image_path_handler.write(os.path.join(label_path.replace("label", "image"), location_info["imagePath"]) + "\n")

    image_width = location_info["imageWidth"]
    image_height = location_info["imageHeight"]

    if len(location_info["shapes"]) < 1:
        continue

    label_line = []
    for shape_info in location_info["shapes"]:
        label_info = shape_info["label"]

        if label_info not in dict_label:
            continue

        label_value = dict_label[label_info]

        [x0, y0], [x1, y1] = shape_info['points']

        # make sure not overflowing
        x0 = min(max(0, round(x0)), image_width)
        x1 = min(max(0, round(x1)), image_width)
        y0 = min(max(0, round(y0)), image_height)
        y1 = min(max(0, round(y1)), image_height)

        # ratio of center coord of rectangle (cx, cy), rectangle width and height compared to image size
        cx = round((x0 + x1) / 2 / image_width, 6)
        cy = round((y0 + y1) / 2 / image_height, 6)
        bw = round(abs(x0 - x1) / image_width, 6)
        bh = round(abs(y0 - y1) / image_height, 6)

        rectangle_value = [label_value, cx, cy, bw, bh]
        rectangle_str = ' '.join(map(lambda x: str(x), rectangle_value)) + '\n'
        label_line.append(rectangle_str)

    label_name = os.path.join(label_path, os.path.splitext(location_info["imagePath"])[0] + ".txt")
    with open(label_name, "wb") as f:
        for str_line in label_line:
            f.write(str_line.encode())

image_path_handler.close()
print("done")



















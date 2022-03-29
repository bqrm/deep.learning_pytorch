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


annotation_path = r"D:/Projects/DataSets/dish_recog_20201116/annotations/"
label_path = r"D:/Projects/DataSets/dish_recog_20201116/labels/"

os.makedirs(label_path, exist_ok=True)

list_filename = []
get_filename_iteratively(annotation_path, list_filename)

for file_idx, annotation_filename in enumerate(list_filename):
    if file_idx % 1000 == 0:
        print("%d: %s" % (file_idx, annotation_filename))

    # if file_idx < 7999:
    #     continue

    read_handler = open(annotation_filename, "rb")
    location_info = json.load(read_handler)
    read_handler.close()

    image_width = location_info["imageWidth"]
    image_height = location_info["imageHeight"]

    if len(location_info["shapes"]) < 1:
        continue

    label_line = []
    dict_label = dict()
    for shape_info in location_info["shapes"]:
        label_info = shape_info["label"]

        if label_info not in dict_label:
            dict_label[label_info] = len(dict_label)

        label_value = dict_label[label_info]

        [x0, y0], [x1, y1] = shape_info['points']

        # make sure not overflowing
        x0 = min(max(0, round(x0)), image_width)
        x1 = min(max(0, round(x1)), image_width)
        y0 = min(max(0, round(y0)), image_height)
        y1 = min(max(0, round(y1)), image_height)

        # ratio of center coord of rectangle (cx, cy), rectangle width and height compared to image size
        cx = (x0 + x1) / 2 / image_width
        cy = (y0 + y1) / 2 / image_height
        bw = abs(x0 - x1) / image_width
        bh = abs(y0 - y1) / image_height

        rectangle_value = [label_value, cx, cy, bw, bh]
        rectangle_str = ' '.join(map(lambda x: str(x), rectangle_value)) + '\n'
        label_line.append(rectangle_str)

    label_name = os.path.join(label_path, location_info["imagePath"] + ".txt")
    with open(label_name, "wb") as f:
        for str_line in label_line:
            f.write(str_line.encode())

print("done")



















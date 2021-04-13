# !/usr/bin/env python
# -*- encoding: utf-8 -*-
# Version:      python 3.6
# Description:  
# Author:       bqrmtao@qq.com
# date:         2021/03/31 23:20

from __future__ import absolute_import

import os
import sys
sys.path.append(os.getcwd())

import base64
import json

from PIL import Image

from utils.common_method import get_filename_iteratively


def convert_txt_to_json(list_image_path, list_label_path):
    if len(list_image_path) != len(list_label_path):
        raise ValueError("")

    for file_idx, (image_path, label_path) in enumerate(zip(list_image_path, list_label_path)):
        print("%d: %s, %s" % (file_idx, image_path, label_path))

        image_basename = os.path.basename(image_path)
        label_basename = os.path.basename(label_path)

        if image_basename not in label_basename:
            raise ValueError("")

        json_data = {
                "version": "3.16.1",
                "flags": {},
            }

        image_handler = Image.open(image_path)

        with open(image_path, "rb") as f:
            str_image_base64 = base64.b64encode(f.read()).decode("ascii")

        with open(label_path, "rb") as f:
            rectangle_info = str(f.read().decode("ascii")).split('\n')

        list_shape = []
        for str_coord in rectangle_info:
            print(str_coord)

            str_value = str_coord.strip(' ').split(' ')

            if len(str_value) != 5:
                continue

            if '0' != str_value[0]:
                raise ValueError("")

            cen_x = float(str_value[1]) * image_handler.width
            cen_y = float(str_value[2]) * image_handler.height
            half_width = float(str_value[3]) * image_handler.width / 2
            half_height = float(str_value[4]) * image_handler.height / 2

            shape_info = {
                "label": "food",
                "points": [
                    [cen_x - half_width, cen_y - half_height],
                    [cen_x + half_width, cen_y + half_height],
                ],
                "shape_type": "rectangle",
                "flags": {}
            }

            list_shape.append(shape_info)
        json_data["shapes"] = list_shape

        json_data["imagePath"] = image_basename
        json_data["imageData"] = str_image_base64
        json_data["imageHeight"] = image_handler.height
        json_data["imageWidth"] = image_handler.width
        json_data["lineColor"] = [0, 255, 0, 128]
        json_data["fillColor"] = [255, 0, 0, 128]

        json_path = os.path.splitext(image_path)[0] + ".json"
        with open(json_path, "wb") as f:
            f.write(json.dumps(json_data, indent=2).encode())


def main():
    image_folder = r"D:/Projects/DataSets/dish_recog/images/"
    label_folder = r"D:/Projects/DataSets/dish_recog/labels/"

    list_image_path = list()
    get_filename_iteratively(image_folder, list_image_path, extension_ignore=["json"])
    list_label_path = list()
    get_filename_iteratively(label_folder, list_label_path)

    convert_txt_to_json(list_image_path, list_label_path)


if "__main__" == __name__:
    main()



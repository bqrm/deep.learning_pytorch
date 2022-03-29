# !/usr/bin/env python
# -*- encoding: utf-8 -*-
# Version:		python 3.6.5
# Description:	
# Author:		bqrmtao@qq.com
# date:			2020/10/22 10:54

from __future__ import absolute_import

import json

from utils.common_method import get_filename_iteratively


folder_name = r"D:/Projects/DataSets/dish_recog/images"

list_filename = []
get_filename_iteratively(folder_name, list_filename, extension_ignore=["png", "jpg", "jpeg"])

for json_idx, json_filename in enumerate(list_filename):
    print("%d: %s" % (json_idx, json_filename))
    with open(json_filename, "rb") as f:
        json_data = json.load(f)

    if "version" in json_data and json_data["version"] != "4.5.6":
        json_data["version"] = "4.5.6"

    for shape_info in json_data["shapes"]:
        try:
            shape_info.pop("line_color")
        except:
            pass
        try:
            shape_info.pop("fill_color")
        except:
            pass

        shape_info["label"] = "food"

        if shape_info["shape_type"] == "polygon":
            print("polygon found")

            coord_x_tl = min(shape_info["points"][0][0], shape_info["points"][1][0], shape_info["points"][2][0], shape_info["points"][3][0])
            coord_x_br = max(shape_info["points"][0][0], shape_info["points"][1][0], shape_info["points"][2][0], shape_info["points"][3][0])
            coord_y_tl = min(shape_info["points"][0][1], shape_info["points"][1][1], shape_info["points"][2][1], shape_info["points"][3][1])
            coord_y_br = max(shape_info["points"][0][1], shape_info["points"][1][1], shape_info["points"][2][1], shape_info["points"][3][1])

            shape_info["points"] = [[coord_x_tl, coord_y_tl], [coord_x_br, coord_y_br]]

            shape_info["shape_type"] = "rectangle"

    with open(json_filename, "wb") as f:
        f.write(json.dumps(json_data, indent=2).encode())



























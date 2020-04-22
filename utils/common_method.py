# -*- coding: utf-8 -*-
# Version:		python 3.6.5
# Description:	
# Author:		bqrmtao@qq.com
# date:			2019/7/10

import os


def delete_recursively(input_path):
    """
    delete file or folder recursively
    :param input_path:
    :return: None
    :except: ValueError if input path not exists.
    """

    if not os.path.exists(input_path):
        raise ValueError("path \"%s\" not exists!" % input_path)

    if os.path.isdir(input_path):
        ls = os.listdir(input_path)
        for i in ls:
            current_path = os.path.join(input_path, i)
            if os.path.isdir(current_path):
                delete_recursively(current_path)
            else:
                os.remove(current_path)
        os.rmdir(input_path)
    else:
        os.remove(input_path)


def get_filename_iteratively(in_path, extension_filter="*", sub_filename=None):
    """
    get file names iteratively
    :param in_path:
    :param extension_filter:
    :param sub_filename:
    :return:
    """
    list_image_paths = []
    for (path, dirs, files) in os.walk(in_path):
        if len(dirs) > 0:
            for sub_folder in dirs:
                get_filename_iteratively(os.path.join(path, sub_folder), extension_filter)
        for filename in files:
            # extension_name = filename.split(".")[1]
            if ("*" == extension_filter or filename.split(".")[1] in extension_filter) and (sub_filename is None or sub_filename in filename):
                list_image_paths.append(os.path.join(path, filename))

    return list_image_paths

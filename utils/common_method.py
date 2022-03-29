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


def get_filename_iteratively(in_path, list_image_path, extension_filter=None, extension_ignore=None, sub_filename=None):
    """
    get file names iteratively
    :param in_path:
    :param list_image_path:
    :param extension_filter:
    :param extension_ignore:
    :param sub_filename:
    :return:
    """

    for (path, dirs, files) in os.walk(in_path):
        if len(dirs) > 0:
            for sub_folder in dirs:
                get_filename_iteratively(os.path.join(path, sub_folder), list_image_path, extension_filter)
        for filename in files:
            ext_name = filename.split('.')[-1]
            if extension_ignore is not None and ext_name in extension_ignore:
                continue
            elif (extension_filter is None or ext_name in extension_filter) and (sub_filename is None or sub_filename in filename):
                list_image_path.append(os.path.join(path, filename))

    return None


def get_leaf_folder(root_path, list_leaf_folder):
    """
    get path of leaf folder
    :param root_path:
    :param list_leaf_folder:
    :return:
    """
    for (path, dirs, files) in os.walk(root_path):
        if len(dirs) == 0:
            list_leaf_folder.append(path)

    return None


def kmgt(byte, unit='g'):
    if 'g' == unit.lower():
        return byte // 1073741824
    elif 'm' == unit.lower():
        return byte // 1048576
    elif 'k' == unit.lower():
        return byte // 1024
    elif 't' == unit.lower():
        return byte // 1099511627776
    else:
        raise ValueError("undefined unit {}".format(unit))







# !/usr/bin/env python
# -*- encoding: utf-8 -*-
# Version:		python 3.6.5
# Description:	
# Author:		bqrmtao@qq.com
# date:			2020/12/3 9:29

from __future__ import absolute_import

import platform


from utils.dict_object import DictToObject


def get_screen_size(without_zoom=True):
    screen_resolution = DictToObject

    if "Windows" == platform.system():
        from win32 import win32api, win32gui, win32print
        from win32.lib import win32con
        from win32api import GetSystemMetrics

        if without_zoom:
            hdc = win32gui.GetDC(0)
            screen_resolution.width = win32print.GetDeviceCaps(hdc, win32con.DESKTOPHORZRES)
            screen_resolution.height = win32print.GetDeviceCaps(hdc, win32con.DESKTOPVERTRES)
        else:
            screen_resolution.width = GetSystemMetrics(0)
            screen_resolution.height = GetSystemMetrics(1)
    elif "Linux" == platform.system():
        screen_resolution.width = 1536
        screen_resolution.height = 864
    else:
        screen_resolution.width = 1536
        screen_resolution.height = 864

    return screen_resolution































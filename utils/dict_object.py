# -*- coding: utf-8 -*-
# Version:		python 3.6.5
# Description:	
# Author:		bqrmtao@qq.com
# date:			2019/12/26


class DictToObject(object):
    def __init__(self):
        pass

    def __setitem__(self, key, value):
        super().__setattr__(key, value)

    def __getitem__(self, item):
        return super().__getattribute__(item)
































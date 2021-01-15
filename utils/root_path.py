# !/usr/bin/env python
# -*- encoding: utf-8 -*-
# Version:		python 3.6.5
# Description:	
# Author:		bqrmtao@qq.com
# date:			2020/11/23 15:01

from __future__ import absolute_import

import os
import sys


class RootPath:
    def __init__(self):
        project_root_path = None

        current_file_path = os.getcwd()

        for path in sys.path:
            if current_file_path == path:
                continue

            if current_file_path.__contains__(path):
                project_root_path = path
                break

        if project_root_path is None:
            # current path is root path if unable to fetch project_root_path
            project_root_path = current_file_path

        # replace \\ with /
        self._root_path = project_root_path.replace("\\", "/")

    def get_root_path(self):
        return self._root_path


root_path = RootPath().get_root_path()


if "__main__" == __name__:
    print(root_path)
































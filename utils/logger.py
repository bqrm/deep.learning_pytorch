# -*- coding: utf-8 -*-
# Version:		python 3.6.5
# Description:	logger in python
# Author:		bqrmtao@qq.com
# date:			2018/4/17

import logging
import os
import time

from utils.root_path import root_path


def set_logger():
    log_path = os.path.join(root_path, "logs")
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(time.strftime("{}\%Y-%m-%d.log".format(log_path), time.localtime(time.time())))
    msg_formatter = logging.Formatter("%(asctime)s\t%(levelname)s\t%(message)s")
    file_handler.setFormatter(msg_formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

logger = logging.getLogger()
set_logger()


if "__main__" == __name__:
    logger.info("aaaaaa")
    logger.debug("bbbbbbbb")

# !/usr/bin/env python
# -*- encoding: utf-8 -*-
# Version:		python 3.6
# Description:	logger in python
# Author:		bqrmtao@qq.com
# date:			2018/4/17

import logging


def logger_config():
    import os
    import time
    from utils.root_path import root_path

    log_path = os.path.join(root_path, "logs", time.strftime("%Y/%m", time.localtime(time.time())))
    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)

    cfg_path = os.path.join(root_path, "cfg", "log.cfg")
    if os.path.exists(cfg_path):
        import configparser
        config_info = configparser.ConfigParser()
        config_info.read(os.path.join(root_path, "cfg", "log.cfg"))

        level_file = config_info.getint("logger", "file")
        level_stream = config_info.getint("logger", "stream")
    else:
        level_file = logging.INFO
        level_stream = logging.INFO

    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(time.strftime("{}/%m%d.log".format(log_path), time.localtime(time.time())))
    msg_formatter = logging.Formatter("%(asctime)s\t%(levelname)s\t%(pathname)s -> %(funcName)s\t%(message)s")
    file_handler.setFormatter(msg_formatter)
    file_handler.setLevel(level_file)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(pathname)s -> %(funcName)s\t: %(message)s"))
    stream_handler.setLevel(level_stream)
    logger.addHandler(stream_handler)


def set_level_logger(level=logging.INFO):
    logger.handlers[0].setLevel(level)


def set_level_stream(level=logging.INFO):
    logger.handlers[1].setLevel(level)


logger = logging.getLogger()
logger_config()


if "__main__" == __name__:
    logger.info("aaaaaa")
    logger.debug("bbbbbbbb")

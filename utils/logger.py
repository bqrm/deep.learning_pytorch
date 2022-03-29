# !/usr/bin/env python
# -*- encoding: utf-8 -*-
# Version:      python 3.6
# Description:  
# Author:       bqrmtao@qq.com
# date:         2021/04/06 13:02
# reference:    https://github.com/cybergrind/safe_logger

from __future__ import absolute_import

import os
# import sys
# sys.path.append(os.getcwd())

# import fcntl
import logging
import logging.handlers
import random
import time
import threading

from utils.root_path import root_path

_lock = threading.RLock()


class TimedRotatingFileHandlerSafe(logging.handlers.TimedRotatingFileHandler):
    def __init__(self, filename, when='midnight', backupCount=30, **kwargs):
        super(TimedRotatingFileHandlerSafe, self).__init__(filename, when=when, backupCount=backupCount, **kwargs)

    def _open(self):
        # if getattr(self, '_lockf', None) and not self._lockf.closed:
        #     return logging.handlers.TimedRotatingFileHandler._open(self)
        with _lock:
            while True:
                try:
                    self._aquire_lock()
                    return logging.handlers.TimedRotatingFileHandler._open(self)
                except (IOError, BlockingIOError):
                    self._release_lock()
                    time.sleep(random.random())
                finally:
                    self._release_lock()

    def _aquire_lock(self):
        try:
            self._lockf = open(self.baseFilename + '_rotating_lock', 'a')
        except PermissionError:
            # name = './{}_rotating_lock'.format(os.path.basename(self.baseFilename))
            # self._lockf = open(name, 'a')
            pass
        # fcntl.flock(self._lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _release_lock(self):
        if not self._lockf.closed:
            # fcntl.lockf(self._lockf, fcntl.LOCK_UN)
            self._lockf.close()

    def is_same_file(self, file1, file2):
        """check is files are same by comparing inodes"""
        return os.fstat(file1.fileno()).st_ino == os.fstat(file2.fileno()).st_ino

    def doRollover(self):
        """
        do a rollover; in this case, a date/time stamp is appended to the filename
        when the rollover happens.  However, you want the file to be named for the
        start of the interval, not the current time.  If there is a backup count,
        then we have to get a list of matching filenames, sort them and remove
        the one with the oldest suffix.
        """
        with _lock:
            return self._inner_rollover()

    def _inner_rollover(self):
        try:
            self._aquire_lock()
        except (IOError, BlockingIOError):
            # cant aquire lock, return
            self._release_lock()
            return

        # get the time that this sequence started at and make it a TimeTuple
        t = self.rolloverAt - self.interval
        if self.utc:
            timeTuple = time.gmtime(t)
        else:
            timeTuple = time.localtime(t)
        dfn = self.baseFilename + "." + time.strftime(self.suffix, timeTuple)

        # check if file is same
        try:
            if self.stream:
                _tmp_f = open(self.baseFilename, 'r')
                is_same = self.is_same_file(self.stream, _tmp_f)
                _tmp_f.close()

                if self.stream:
                    self.stream.close()
                if is_same and not os.path.exists(dfn):
                    os.rename(self.baseFilename, dfn)
        except ValueError:
            # ValueError: I/O operation on closed file
            is_same = False

        if self.backupCount > 0:
            for s in self.getFilesToDelete():
                os.remove(s)
        self.mode = 'a'
        self.stream = self._open()
        currentTime = int(time.time())
        newRolloverAt = self.computeRollover(currentTime)
        while newRolloverAt <= currentTime:
            newRolloverAt = newRolloverAt + self.interval
        #If DST changes and midnight or weekly rollover, adjust for this.
        if (self.when == 'MIDNIGHT' or self.when.startswith('W')) and not self.utc:
            dstNow = time.localtime(currentTime)[-1]
            dstAtRollover = time.localtime(newRolloverAt)[-1]
            if dstNow != dstAtRollover:
                if not dstNow:  # DST kicks in before next rollover, so we need to deduct an hour
                    newRolloverAt = newRolloverAt - 3600
                else:           # DST bows out before next rollover, so we need to add an hour
                    newRolloverAt = newRolloverAt + 3600
        self.rolloverAt = newRolloverAt
        self._release_lock()


class NullHandler(logging.Handler):
    def emit(self, record):
        pass
    def write(self, *args, **kwargs):
        pass


def logger_config():
    log_path = os.path.join(root_path, "logs", time.strftime("%Y/%m", time.localtime(time.time())))
    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)
    
    logging.basicConfig(level=logging.DEBUG, stream=NullHandler())
    
    log_file_path = time.strftime("{}/%d.log".format(log_path), time.localtime(time.time()))
    log_handler = TimedRotatingFileHandlerSafe(log_file_path, when='MIDNIGHT')
    log_handler.setLevel(logging.INFO)
    log_handler.setFormatter(logging.Formatter("%(asctime)s\t%(levelname)s\t%(pathname)s -> %(funcName)s\t%(message)s"))
    logger.addHandler(log_handler)
    
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(asctime)s\t%(pathname)s -> %(funcName)s\t: %(message)s"))
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)
    
    # ERR_FILE = './logs/error.log'
    # err_handler = TimedRotatingFileHandlerSafe(ERR_FILE, when='MIDNIGHT')
    # err_handler.setLevel(logging.ERROR)
    # err_handler.setFormatter(FORMATTER)
    # root.addHandler(err_handler)


def set_level_logger(level=logging.INFO):
    logger.handlers[1].setLevel(level)


def set_level_stream(level=logging.INFO):
    logger.handlers[2].setLevel(level)


logger = logging.root
logger_config()


if "__main__" == __name__:
    # lg = logging.getLogger('testme')
    
    iter_round = 0
    while True:
        logger.debug(iter_round)
        logger.error(iter_round)
        time.sleep(0.5)

        if iter_round % 20 == 0:
            iter_round = 0
            set_level_logger(logging.DEBUG)
            set_level_stream(logging.DEBUG)
        elif iter_round % 10 == 0:
            set_level_logger(logging.INFO)
            set_level_stream(logging.INFO)
        
        iter_round += 1











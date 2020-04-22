# -*- coding: utf-8 -*-
# Version:		python 3.6.5
# Description:	logger in python
# Author:		bqrmtao@qq.com
# date:			2018/4/17

import logging
import os
import time

if not os.path.exists("logs"):
    os.mkdir("logs")

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s.%(msecs)03d %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    filename=time.strftime('logs\%Y-%m-%d', time.localtime(time.time())) + '.log',
                    filemode='a')

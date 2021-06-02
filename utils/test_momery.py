# !/usr/bin/env python
# -*- encoding: utf-8 -*-
# Version:      python 3.6
# Description:  
# Author:       bqrmtao@qq.com
# date:         2021/04/14 15:50

from __future__ import absolute_import


if "__main__" == __name__:
    import pynvml

    pynvml.nvmlInit()
    num_gpu = pynvml.nvmlDeviceGetCount()
    num_machine = pynvml.nvmlUnitGetCount()

    print()
    print("{} gpu(s) in total\n".format(num_gpu))
    print("memory on gpu:")
    print(" id |     total    |     free     |      used    |       name")

    mb_inbyte = 1048576

    for gpu_idx in range(num_gpu):
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)   # gpu_id
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)

        print(" {:2d} | {:7.2f} | {:7.2f} | {:7.2f} | {}".format(
            gpu_idx,
            meminfo.total / mb_inbyte,
            meminfo.free / mb_inbyte,
            meminfo.used / mb_inbyte,
            pynvml.nvmlDeviceGetName(handle).decode())
        )
















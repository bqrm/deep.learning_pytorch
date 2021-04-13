# -*- coding: utf-8 -*-
# Version:		python 3.6.5
# Description:	
# Author:		bqrmtao@qq.com
# date:			2020/3/2 22:11

# https://www.cnblogs.com/hizhaolei/p/11226146.html
# https://zhuanlan.zhihu.com/p/38056115
# https://cloud.tencent.com/developer/article/1507565

from __future__ import absolute_import

import glob
import os
import torch

from utils.common_method import delete_recursively
from utils.common_method import get_filename_iteratively


class ModelSaver(object):
    def __init__(self, model_name, root_path="./", maximum_checkpoint=5):
        self.checkpoint_dir = os.path.join(root_path, model_name)
        self.maximum_checkpoint = maximum_checkpoint

        self.runs = get_filename_iteratively(self.checkpoint_dir)
        while len(self.runs) > self.maximum_checkpoint:
            os.remove(self.runs[0])
            self.runs.pop(0)

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

    def num_checkpoint(self):
        return len(self.runs)

    def get_checkpoint(self, index=-1):
        if len(self.runs) > 0:
            return torch.load(self.runs[index], map_location=lambda storage, loc: storage)
        else:
            return None

    def save_checkpoint(self, state):
        run_id = int(os.path.basename(self.runs[-1]).split('.')[0]) + 1 if self.runs else 0
        checkpoint_name = os.path.join(self.checkpoint_dir, "%03d.pth" % run_id)

        torch.save(state, checkpoint_name)
        self.runs.append(checkpoint_name)
        
        while len(self.runs) > self.maximum_checkpoint:
            os.remove(self.runs[0])
            self.runs.pop(0)

    def save_model_config(self):
        pass





























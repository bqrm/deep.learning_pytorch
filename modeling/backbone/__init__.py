# -*- coding: utf-8 -*-
# Version:		python 3.6.5
# Description:	
# Author:		bqrmtao@qq.com
# date:			2019/12/13

from modeling.classification.vgg import vgg_backbone


def build_backbone(backbone, **kwargs):
    if backbone == "vgg":
        return vgg_backbone(**kwargs)
    else:
        raise NotImplementedError


















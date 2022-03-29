# -*- coding: utf-8 -*-
# Version:		python 3.6.5
# Description:	
# Author:		bqrmtao@qq.com
# date:			2020/11/21 14:47

from __future__ import absolute_import

import os
import threading

from configparser import RawConfigParser

class BaseConfig(RawConfigParser):
    def __init__(self, config_filename, defaults=None):
        super(BaseConfig, self).__init__(defaults=defaults)
        
        self._config_filename = os.path.realpath(config_filename)

        if not os.path.exists(self._config_filename):
            raise FileNotFoundError("file \"%s\" not found" % self._config_filename)

        self.read(self._config_filename, encoding="utf-8")

    def get(self, section_name, item_name, strip_blank=True, strip_quote=True):
        str_info = super().get(section_name, item_name)
        if strip_blank:
            str_info = str_info.strip()
        if strip_quote:
            str_info = str_info.strip('"').strip("'")

        return str_info
    
    def get_section(self, section_name):
        keys = self.options(section_name)

        return {
            key: self.get(section_name, key)
            for key in keys
        }

    def optionxform(self, optionstr):
        """
        optionxform in parent class transfer capital into lower case, disable it
        """
        return optionstr



























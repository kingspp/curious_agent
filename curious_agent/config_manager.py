# -*- coding: utf-8 -*-
"""
| **@created on:** 09/08/17,
| **@author:** prathyushsp,
| **@version:** v0.0.1
|
| **Description:**
| 
|
| **Sphinx Documentation Status:** Complete
|
..todo::
    --
"""

from typeguard import typechecked
from collections import OrderedDict
import os
from typing import Union
import logging
import json
from . import setup_logging
from curious_agent import Singleton
from curious_agent import string_constants as constants
from curious_agent import MODULE_PATH

logger = logging.getLogger(__name__)


class BaseConfig(object):
    """
    | **@author:** Prathyush SP
    | Common Configuration
    """

    # todo: Prathyush SP: Convert keys to string constants
    @typechecked
    def __init__(self, base_config: dict):
        try:

            self.LOG_LEVEL = base_config['log_level']
            self.BASE_DIR = MODULE_PATH+'/../experiments' if base_config['base_dir'] == "" else base_config['base_dir']
            self.EXPERIMENTS_META_NAME = base_config['experiments_meta_name']
            self.PATH_LOG = base_config['path']['logs']
            self.PATH_CHECKPOINT = base_config['path']['checkpoint']
            self.PATH_GRAPHS = base_config['path']['graphs']
            self.PATH_CONFIGS= base_config['path']['configs']
            self._GLOBAL_LOGGING_CONFIG_FILE_PATH = os.path.join("/".join(__file__.split('/')[:-1]), 'config',
                                                                 'module_logging.yaml')
            self.TENSORBOARD_SUMMARIES = base_config['tensorboard_summaries']
            self.TENSORBOARD_PORT = base_config['tensorboard_port']
            self.PYTHON_OPTIMISE = base_config['python_optimise']
            self.DRY_RUN = base_config['dry_run']
            os.environ['PYTHONOPTIMIZE'] = str(self.PYTHON_OPTIMISE)
        except KeyError as ke:
            raise Exception('Key Error. Config Error', ke)


class ConfigManager(metaclass=Singleton):
    """
    | **@author:** Prathyush SP
    |
    | DL Configuration Manager
    """

    @typechecked
    def __init__(self, module_config: OrderedDict):
        # todo: Test Support for multiple dl frameworks
        try:
            self.BaseConfig = BaseConfig(MODULE_CONFIG_DATA['base_config'])
        except KeyError as ke:
            raise Exception('Key not found. ', ke)

    def get_config_manager(self):
        """
        | **@author:** Prathyush SP
        |
        | Get Configuration Manager
        :return: Configuration Manager
        """
        return self

    @typechecked
    def update_config_manager(self, config_file_path: str):
        """
        | **@author:** Prathyush SP
        |
        | Update Configuration Manager
        :param config_file_path: Configuration file path
        """
        logger.info("Updating Module Configuration - Config File: {}".format(config_file_path))
        self.__init__(config_file_path=config_file_path)

    def update_logging_config_manager(self, config_file_path: str):
        """
        | **@author:** Prathyush SP
        |
        | Update Logging Configuration
        :param config_file_path: Configuration file path
        """
        logger.info("Updating Library Logging configuration - Config File:{}".format(config_file_path))
        setup_logging(default_path=config_file_path)
        self.BaseConfig._GLOBAL_LOGGING_CONFIG_FILE_PATH = config_file_path


ConfigPath = os.path.join("/".join(__file__.split('/')[:-1]), 'config', 'base_config.json')
try:
    MODULE_CONFIG_DATA = json.load(open(ConfigPath), object_pairs_hook=OrderedDict)
except Exception as e:
    logger.critical(
        'Configuration file path error. Please provide configuration file path: {}'.format(ConfigPath))
    raise Exception(
        'Configuration file path error. Please provide configuration file path: ' + ConfigPath, e)

MODULE_CONFIG = ConfigManager(module_config=MODULE_CONFIG_DATA).get_config_manager()


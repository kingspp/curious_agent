#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch.nn as nn
from abc import ABCMeta, abstractmethod
import os
import torch
import logging
from curious_agent.util import generate_uuid

logger = logging.getLogger(__name__)


class Model(nn.Module, metaclass=ABCMeta):
    """Model Abstract Class
    """

    def __init__(self, env, args, name: str = ''):
        """
        You can add additional arguments as you need.
        In the constructor we instantiate modules and assign them as
        member variables.
        """
        super(Model, self).__init__()
        self.name = name
        self.env=env
        self.args = args

    @abstractmethod
    def forward(self, inp):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        pass

    def save(self, file_name_with_path):
        """
        Save Model based on condition
        :param i_episode: Episode Number
        """
        with open(file_name_with_path, 'wb') as f:
            torch.save(self, f)
        logger.info(f"{self.name} saved successfully at {file_name_with_path}")

    def load(self, file_name_with_path, device=""):
        """
        Load Model
        :return:
        """
        logger.info(f"Restoring {self.name} model from {self.args.load_dir} . . . ")
        model = torch.load(file_name_with_path,
                           map_location=torch.device(device)).to(device) if device != "" else None
        logger.info(f"{self.name} Model successfully restored.")
        return model

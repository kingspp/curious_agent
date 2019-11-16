#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
from abc import ABCMeta, abstractmethod


class Model(nn.Module, metaclass=ABCMeta):
    """Model Abstract Class
    """

    def __init__(self, env, args):
        """
        You can add additional arguments as you need.
        In the constructor we instantiate modules and assign them as
        member variables.
        """
        super(Model, self).__init__()

    @abstractmethod
    def forward(self, inp):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        pass

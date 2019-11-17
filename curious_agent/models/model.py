#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch.nn as nn
from abc import ABCMeta, abstractmethod
import os
import torch


class Model(nn.Module, metaclass=ABCMeta):
    """Model Abstract Class
    """

    def __init__(self, name: str, env, args):
        """
        You can add additional arguments as you need.
        In the constructor we instantiate modules and assign them as
        member variables.
        """
        super(Model, self).__init__()
        self.name = name

    @abstractmethod
    def forward(self, inp):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        pass

    def save_model(self, i_episode):
        """
        Save Model based on condition
        :param i_episode: Episode Number
        """

        model_file = os.path.join(self.args.save_dir, f'model_e{i_episode}.th')
        with open(model_file, 'wb') as f:
            torch.save(self.policy_net, f)
        print(f"{self.model} saved successfully at {model_file}")

    def load_model(self):
        """
        Load Model
        :return:
        """
        print(f"Restoring {self.name} model from {self.args.load_dir} . . . ")
        model = torch.load(self.args.load_dir,
                           map_location=torch.device(self.args.device)).to(self.args.device)
        print(f"{self.name} Model successfully restored.")
        return model

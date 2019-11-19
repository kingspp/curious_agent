#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
from curious_agent.models import Model


class CNNModel(Model):
    """Initialize a deep Q-learning network
    
    Hints:
    -----
        Original paper for DQN
    https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
    
    This is just a hint. You can build your own structure.
    """

    def __init__(self, env, args):
        """
        You can add additional arguments as you need.
        In the constructor we instantiate modules and assign them as
        member variables.
        """
        super(CNNModel, self).__init__(env=env, args=args)
        self.args = args
        self.num_actions = env.action_space.n
        self.head = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.BatchNorm2d(32) if self.args.use_bnorm else nn.Identity(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(64) if self.args.use_bnorm else nn.Identity(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(64) if self.args.use_bnorm else nn.Identity()
        )
        self.tail = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions),
            nn.Softmax()
        )

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        x = self.head(x)
        x = self.tail(x.view(x.size(0), -1))
        return x

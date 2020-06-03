# -*- coding: utf-8 -*-
"""
@created on: 12/6/19,
@author: Shreesha N,
@version: v0.0.1
@system name: badgod
Description:

..todo::

"""

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from functools import reduce
from torch import tensor
from curious_agent.models.model import Model


class Flatten(Model):
    def forward(self, input):
        return input.view(input.size(0), -1)


class ICM(Model):

    def __init__(self, num_actions, args, env):
        super(ICM, self).__init__(args=args, env=env)
        self.num_actions = num_actions

        self.feature_extractor = nn.Sequential(
                nn.Conv2d(4, 32, 8, 4),
                nn.LeakyReLU(),
                nn.Conv2d(32, 64, 4, 2),
                nn.LeakyReLU(),
                nn.Conv2d(64, 64, 3, 1),
                nn.LeakyReLU(),
                Flatten(args={}, env={}),
                nn.Linear(7 * 7 * 64, 512),
        )

        self.inverse_model = nn.Sequential(
                nn.Linear(512 * 2, 512),
                nn.LeakyReLU(),
                nn.Linear(512, num_actions)
        )

        self.forward_model = nn.Sequential(
                nn.Linear(512 + num_actions, 512),
                nn.LeakyReLU(),
                nn.Linear(512, 512)
        )

    def forward(self, inp):
        state_batch, next_state_batch, onehot_action_batch = inp
        encoded_current_state_batch = self.feature_extractor(state_batch)  # phi S
        encoded_next_state_batch = self.feature_extractor(next_state_batch)  # phi S+1
        self.encoded_next_state_batch_shape = reduce(lambda x, y: x * y, encoded_next_state_batch.shape[1:])

        forward_model_inp = torch.cat((encoded_current_state_batch, onehot_action_batch), dim=1)
        predicted_next_state = self.forward_model(forward_model_inp)  # predicted Phi s+1

        inverse_model_inp = torch.cat((encoded_current_state_batch, encoded_next_state_batch), dim=1)
        predicted_action = self.inverse_model(inverse_model_inp)  # pred action

        return encoded_next_state_batch, predicted_next_state, predicted_action

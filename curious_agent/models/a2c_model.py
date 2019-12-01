#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from curious_agent.models import Model

class A2CModel(Model):
    def __init__(self, lr, input_dims, n_actions):
        super(A2CModel, self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            # nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.policy = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 4)
        )
        self.v = nn.Sequential(
            nn.Linear(7 * 7 * 64, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, observation):
        state = preprocess(observation)
        x = self.conv(state)
        x = x.view(x.size(0), -1)
        policy = self.policy(x)
        v = self.v(x)
        return (policy, v)

    def preprocess(self, observation):
        return observation
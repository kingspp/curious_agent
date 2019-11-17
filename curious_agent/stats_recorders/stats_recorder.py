# -*- coding: utf-8 -*-
"""
.. module:: stats_recorder
    :platform: Linux
    :synopsis: module that contains the definition of the abstract stats_recorder class that collects and saves stats on
    the disk, while also saving a video for intuitive human performance inspection purposes

.. moduleauthor:: Sinan Morcel <shmorcel@wpi.edu>
"""

from curious_agent.environments.environment import Environment
from curious_agent.agents.agent import Agent


class StatsRecorder(object):
    def __init__(self, agent: Agent, env: Environment):
        self.agent = agent
        self.env = env

    def load(self, location):
        raise NotImplementedError

    def record(self, output):
        raise NotImplementedError

# -*- coding: utf-8 -*-
"""
.. module:: visualizer
    :platform: Linux
    :synopsis: module that contains the definition of the state visualizer for human auditing of agent performance purposes

.. moduleauthor:: Sinan Morcel <shmorcel@wpi.edu>
"""


from typeguard import typechecked
import numpy as np


class Visualizer:
    """Class defining shared functionality that environments have to implement to visualize the state

    This class is meant to define an interface that accepts OpenAI state-types.

    """
    @typechecked
    def show_state(self, state: np.ndarray):
        """A method that displays the state for the purpose of agent-performance auditing.

        :param state: an n-dimensional numpy array that contains the state information, often directly returned by the
        an OpenAI environment.
        :return: None
        """
        raise NotImplementedError

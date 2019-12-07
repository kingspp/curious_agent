# -*- coding: utf-8 -*-
"""
@created on: 12/6/19,
@author: Shreesha N,
@version: v0.0.1
@system name: badgod
Description:

..todo::

"""


def generate_onehot(index, num_actions):
    return [1 if i == index else 0 for i, x in enumerate(range(num_actions))]

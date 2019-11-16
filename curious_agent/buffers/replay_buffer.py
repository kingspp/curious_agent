from collections import namedtuple
from torch.autograd import Variable
import torch
import random


class ReplayBuffer(object):
    """ Facilitates memory replay. """

    def __init__(self, capacity, args):
        self.capacity = capacity
        self.memory = []
        self.idx = 0
        self.args = args
        self.transition = namedtuple('Transition',
                                     ('state', 'action', 'next_state', 'reward', 'done'))

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.idx] = self.transition(*args)
        self.idx = (self.idx + 1) % self.capacity

    def sample(self, bsz):
        batch = random.sample(self.memory, bsz)
        return map(lambda x: Variable(torch.cat(x, 0)).to(self.args.device), zip(*batch))

import numpy as np
from collections import deque, namedtuple
from torch.autograd import Variable
import torch


class PrioritizedBuffer(object):
    def __init__(self, capacity, args, prob_alpha=0.6):
        self.prob_alpha = prob_alpha
        self.capacity = capacity
        self.memory = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.transition = namedtuple('Transition',
                                     ('state', 'action', 'next_state', 'reward', 'done'))
        self.args = args

    def push(self, *args):
        max_prio = self.priorities.max() if self.memory else 1.0

        if len(self.memory) < self.capacity:
            self.memory.append(self.transition(*args))
        else:
            self.memory[self.pos] = self.transition(*args)

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.memory) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        probs = prios ** self.prob_alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        samples = [self.memory[idx] for idx in indices]

        total = len(self.memory)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)
        return [*map(lambda x: torch.Variable(torch.cat(x, 0)).to(self.args.device), zip(*samples)), indices, weights]

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.memory)

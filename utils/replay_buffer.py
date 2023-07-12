import torch
import numpy as np
import random
from collections import deque


class ReplayBuffer:
    """ 经验回放池, 每次采样 batch_size 个样本 """
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.buffer = deque(maxlen=self.capacity)

    def push(self, transitions):
        """_summary_
            :param transitions: (tuple)
        """
        self.buffer.append(transitions)

    def sample(self, batch_size: int = None, sequential: bool = True):
        if batch_size is None or batch_size > len(self.buffer):
            batch_size = len(self.buffer)

        if sequential:  # sequential sampling
            rand = random.randint(0, len(self.buffer) - batch_size)
            batch = [self.buffer[i] for i in range(rand, rand + batch_size)]
        else:
            batch = random.sample(self.buffer, batch_size)

        s, a, a_logprob, s_, r, dw, done = zip(*batch)

        s = torch.tensor(np.asarray(s), dtype=torch.float)
        a = torch.tensor(np.asarray(a), dtype=torch.float)
        a_logprob = torch.tensor(np.asarray(a_logprob), dtype=torch.float)
        s_ = torch.tensor(np.asarray(s_), dtype=torch.float)
        r = torch.tensor(np.asarray(r).reshape((self.capacity, 1)), dtype=torch.float)
        dw = torch.tensor(np.asarray(dw).reshape((self.capacity, 1)), dtype=torch.float)
        done = torch.tensor(np.asarray(done).reshape((self.capacity, 1)), dtype=torch.float)

        return s, a, a_logprob, s_, r, dw, done

    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)

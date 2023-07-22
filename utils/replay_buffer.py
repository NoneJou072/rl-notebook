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
        """ 存储transition到经验回放中
            :param transitions: (tuple)
        """
        self.buffer.append(transitions)

    def sample(self, batch_size: int = None, sequential: bool = True, with_log=True):
        # 如果批量大小大于经验回放的容量，则取经验回放的容量
        if batch_size is None or batch_size > len(self.buffer):
            batch_size = len(self.buffer)

        if sequential:  # 顺序采样 sequential sampling
            rand = random.randint(0, len(self.buffer) - batch_size)
            batch = [self.buffer[i] for i in range(rand, rand + batch_size)]
        else:  # 随机采样
            batch = random.sample(self.buffer, batch_size)

        a_logprob = None
        if with_log:
            s, a, a_logprob, s_, r, dw, done = zip(*batch)
        else:
            s, a, s_, r, dw, done = zip(*batch)

        s = torch.tensor(np.asarray(s), dtype=torch.float)
        a = torch.tensor(np.asarray(a), dtype=torch.float)
        if with_log:
            a_logprob = torch.tensor(np.asarray(a_logprob), dtype=torch.float)
        s_ = torch.tensor(np.asarray(s_), dtype=torch.float)
        r = torch.tensor(np.asarray(r), dtype=torch.float).view(batch_size, 1)
        dw = torch.tensor(np.asarray(dw), dtype=torch.float).view(batch_size, 1)
        done = torch.tensor(np.asarray(done), dtype=torch.float).view(batch_size, 1)
        if with_log:
            return s, a, a_logprob, s_, r, dw, done
        else:
            return s, a, s_, r, dw, done

    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)


class ReplayBufferDiscreteAction(ReplayBuffer):
    def sample(self, batch_size: int = None, sequential: bool = True, with_log=True):
        # 如果批量大小大于经验回放的容量，则取经验回放的容量
        if batch_size is None or batch_size > len(self.buffer):
            batch_size = len(self.buffer)

        if sequential:  # 顺序采样 sequential sampling
            rand = random.randint(0, len(self.buffer) - batch_size)
            batch = [self.buffer[i] for i in range(rand, rand + batch_size)]
        else:  # 随机采样
            batch = random.sample(self.buffer, batch_size)

        a_logprob = None
        if with_log:
            s, a, a_logprob, s_, r, dw, done = zip(*batch)
        else:
            s, a, s_, r, dw, done = zip(*batch)

        s = torch.tensor(np.asarray(s), dtype=torch.float)
        a = torch.tensor(np.asarray(a))
        if with_log:
            a_logprob = torch.tensor(np.asarray(a_logprob), dtype=torch.float)
        s_ = torch.tensor(np.asarray(s_), dtype=torch.float)
        r = torch.tensor(np.asarray(r).reshape((batch_size, 1)), dtype=torch.float)
        dw = torch.tensor(np.asarray(dw).reshape((batch_size, 1)), dtype=torch.float)
        done = torch.tensor(np.asarray(done).reshape((batch_size, 1)), dtype=torch.float)
        if with_log:
            return s, a, a_logprob, s_, r, dw, done
        else:
            return s, a, s_, r, dw, done

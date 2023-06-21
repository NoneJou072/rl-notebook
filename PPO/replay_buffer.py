import torch
import numpy as np
import random
from collections import deque


class ReplayBufferQue:
    """ DQN的经验回放池, 每次采样 batch_size 个样本 """
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.buffer = deque(maxlen=self.capacity)

    def push(self, transitions):
        """_summary_
            :param transitions: (tuple)
        """
        self.buffer.append(transitions)

    def sample(self, batch_size: int, sequential: bool = False):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        if sequential:  # sequential sampling
            rand = random.randint(0, len(self.buffer) - batch_size)
            batch = [self.buffer[i] for i in range(rand, rand + batch_size)]
            return zip(*batch)
        else:
            batch = random.sample(self.buffer, batch_size)
            return zip(*batch)

    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)


class PGReplay(ReplayBufferQue):
    """PG的经验回放池, 每次采样所有样本, 因此只需要继承ReplayBufferQue, 重写sample方法即可
    """
    def __init__(self, capacity: int):
        super(PGReplay, self).__init__(capacity)

    def sample(self):
        """ sample all the transitions """
        batch = list(self.buffer)
        return zip(*batch)

    def sample_tensor(self, device):
        s, a, a_logprob, s_, r, dw, done = self.sample()
        s = torch.tensor(np.asarray(s), dtype=torch.float)
        a = torch.tensor(np.asarray(a), dtype=torch.float)
        a_logprob = torch.tensor(np.asarray(a_logprob), dtype=torch.float)
        s_ = torch.tensor(np.asarray(s_), dtype=torch.float)
        r = torch.tensor(np.asarray(r).reshape((self.capacity, 1)), dtype=torch.float)
        dw = torch.tensor(np.asarray(dw).reshape((self.capacity, 1)), dtype=torch.float)
        done = torch.tensor(np.asarray(done).reshape((self.capacity, 1)), dtype=torch.float)

        return s, a, a_logprob, s_, r, dw, done


class ReplayBuffer:
    def __init__(self, args):
        self.buffer_size = args.buffer_size
        self.s = deque(maxlen=args.buffer_size)
        self.a = deque(maxlen=args.buffer_size)
        self.a_logprob = deque(maxlen=args.buffer_size)
        self.r = deque(maxlen=args.buffer_size)
        self.s_ = deque(maxlen=args.buffer_size)
        self.dw = deque(maxlen=args.buffer_size)
        self.done = deque(maxlen=args.buffer_size)

        # self.a = np.zeros((args.buffer_size, args.n_actions))
        # self.a_logprob = np.zeros((args.buffer_size, args.n_actions))
        # self.s_ = np.zeros((args.buffer_size, args.n_states))

    def store(self, s, a, a_logprob, r, s_, dw, done):
        self.s.append(s)
        self.a.append(a)
        self.a_logprob.append(a_logprob)
        self.r.append(r)
        self.s_.append(s_)
        self.dw.append(dw)
        self.done.append(done)

    def numpy_to_tensor(self):
        s = torch.tensor(np.array(self.s), dtype=torch.float)
        a = torch.tensor(np.array(self.a), dtype=torch.float)
        a_logprob = torch.tensor(np.array(self.a_logprob), dtype=torch.float)
        r = torch.tensor(np.array(self.r).reshape((self.buffer_size, 1)), dtype=torch.float)
        s_ = torch.tensor(np.array(self.s_), dtype=torch.float)
        dw = torch.tensor(np.array(self.dw).reshape((self.buffer_size, 1)), dtype=torch.float)
        done = torch.tensor(np.array(self.done).reshape((self.buffer_size, 1)), dtype=torch.float)

        return s, a, a_logprob, r, s_, dw, done

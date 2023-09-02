import torch
import numpy as np
import random
from collections import deque


class ReplayBuffer:
    """ 经验回放池, 用于存储transition, 然后随机采样transition用于训练 """

    def __init__(self, capacity: int) -> None:
        """ 初始化经验回放池

            :param capacity: (int) 经验回放池的容量
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=self.capacity)

    def push(self, transitions: tuple):
        """ 存储transition到经验回放中

            :param transitions: (tuple) transition
        """
        self.buffer.append(transitions)

    def sample(self, batch_size: int = None, sequential: bool = True, with_log=True):
        """ 采样transition, 每次采样 batch_size 个样本

            :param batch_size: (int) 批量大小
            :param sequential: (bool) 是否顺序采样
            :param with_log: (bool) 是否返回动作的对数概率
            :return: (tuple)
        """
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
    """ 经验回放池, 用于存储transition, 然后随机采样transition用于训练 """

    def __init__(self, capacity: int, k_future: int) -> None:
        """ 初始化经验回放池

            :param capacity: (int) 经验回放池的容量
        """
        super().__init__(capacity)
        self.k_future = k_future

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
        a = torch.tensor(np.asarray(a), dtype=torch.int64)
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


class Trajectory:
    """ 用于存储一个完整的轨迹 """

    def __init__(self) -> None:
        self.buffer = []

    def push(self, transitions: tuple):
        """ 存储transition到经验回放中

            :param transitions: (tuple)
        """
        self.buffer.append(transitions)

    def __len__(self):
        return len(self.buffer)


class HERReplayBuffer(ReplayBuffer):
    """ Hindisght Experience Replay Buffer """

    def __init__(self, capacity: int, k_future: int, env) -> None:
        super().__init__(capacity)
        self.env = env  # 需要调用 compute_reward 函数
        self.k_future = k_future
        self.her_ratio = 0.8

    def push(self, trajectory: Trajectory):
        """ 存储 trajectory 到经验回放中

            :param trajectory: (Trajectory)
        """
        self.buffer.append(trajectory)

    def sample(self, batch_size: int = None, sequential: bool = True, with_log=True):
        batch = []
        # 如果批量大小大于经验回放的容量，则取经验回放的容量
        if batch_size is None or batch_size > len(self.buffer):
            batch_size = len(self.buffer)

        for _ in range(batch_size):
            # 从经验回放中随机采样一个轨迹
            traj = random.sample(self.buffer, 1)[0]
            # 随机从轨迹中采样一个状态作为当前状态
            step_state = np.random.randint(0, len(traj) - 1)
            s, a, s_, r, dw, done, achieved_g, desired_g = traj.buffer[step_state]

            if np.random.uniform() < self.her_ratio:
                # 从回合当前状态的步数开始向后随机采样 k 个步数
                step_goal = np.random.randint(step_state + 1, len(traj))
                s, a, s_, r, dw, done, achieved_g, desired_g = traj.buffer[step_goal]
                r = self.env.compute_reward(achieved_g, desired_g, None)

            batch.append((s, a, s_, r, dw, done, achieved_g, desired_g))

        s, a, s_, r, dw, done, _, g = zip(*batch)

        s = torch.tensor(np.asarray(s), dtype=torch.float)
        a = torch.tensor(np.asarray(a), dtype=torch.float)
        s_ = torch.tensor(np.asarray(s_), dtype=torch.float)
        r = torch.tensor(np.asarray(r), dtype=torch.float).view(-1, 1)
        dw = torch.tensor(np.asarray(dw), dtype=torch.float).view(-1, 1)
        done = torch.tensor(np.asarray(done), dtype=torch.float).view(-1, 1)
        g = torch.tensor(np.asarray(g), dtype=torch.float)

        return s, a, s_, r, dw, done, g

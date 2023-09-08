import torch
import numpy as np
import random
from collections import deque
from copy import deepcopy as dc


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

    def __init__(self, capacity: int) -> None:
        """ 初始化经验回放池

            :param capacity: (int) 经验回放池的容量
        """
        super().__init__(capacity)

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
        self.future_p = 1 - (1. / (1 + k_future))

    def push(self, trajectory: Trajectory):
        """ 存储 trajectory 到经验回放中

            :param trajectory: (Trajectory)
        """
        self.buffer.append(trajectory)

    def sample(self, batch_size: int = 256, sequential: bool = True, with_log=True, device='cpu'):
        # 随机取 batch size 个回合索引和时间步索引
        ep_indices = np.random.randint(0, len(self.buffer), batch_size)
        time_indices = np.random.randint(0, len(self.buffer[0]), batch_size)
        states = []
        actions = []
        desired_goals = []
        next_states = []
        next_achieved_goals = []

        # 取出对应回合与时间步的 transitions
        for episode, timestep in zip(ep_indices, time_indices):
            states.append(dc(self.buffer[episode].buffer[timestep][0]))
            actions.append(dc(self.buffer[episode].buffer[timestep][1]))
            next_states.append(dc(self.buffer[episode].buffer[timestep][2]))
            desired_goals.append(dc(self.buffer[episode].buffer[timestep][5]))
            next_achieved_goals.append(dc(self.buffer[episode].buffer[timestep][6]))

        # 将列表升维并转换成数组类型
        states = np.vstack(states)
        actions = np.vstack(actions)
        desired_goals = np.vstack(desired_goals)
        next_achieved_goals = np.vstack(next_achieved_goals)
        next_states = np.vstack(next_states)

        # 根据 future k 概率随机选择要在批量中替换的索引
        her_indices = np.where(np.random.uniform(size=batch_size) < self.future_p)
        future_offset = np.random.uniform(size=batch_size) * (len(self.buffer[0]) - time_indices)
        future_offset = future_offset.astype(int)
        future_t = (time_indices + future_offset)[her_indices]

        future_ag = []
        for epi, f_offset in zip(ep_indices[her_indices], future_t):
            future_ag.append(dc(self.buffer[epi].buffer[f_offset][4]))
        future_ag = np.vstack(future_ag)

        desired_goals[her_indices] = future_ag
        rewards = np.expand_dims(self.env.unwrapped.compute_reward(next_achieved_goals, desired_goals, None), 1)

        s = torch.tensor(np.asarray(states), dtype=torch.float).to(device)
        a = torch.tensor(np.asarray(actions), dtype=torch.float).to(device)
        s_ = torch.tensor(np.asarray(next_states), dtype=torch.float).to(device)
        r = torch.tensor(np.asarray(rewards), dtype=torch.float).to(device)
        g = torch.tensor(np.asarray(desired_goals), dtype=torch.float).to(device)

        return s, a, s_, r, g

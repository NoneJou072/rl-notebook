from utils.replay_buffer import ReplayBuffer, Trajectory
import numpy as np
import torch
from copy import deepcopy as dc


class RHERReplayBuffer(ReplayBuffer):
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

    def sample(self, batch_size: int = 256, sequential: bool = True, with_log=True, device='cpu', task='reach'):
        """ 从经验回放中随机采样 batch_size 个 transitions

        :param batch_size: (int) 采样的 batch size
        :param sequential: (bool)
        :param with_log: (bool)
        :param device: (str) cpu or cuda
        :param task: (str) reach, drawer, place
        :return: (torch.tensor) s, a, s_, r, g
        """
        ep_indices = np.random.randint(0, len(self.buffer), batch_size)

        time_indices = []
        for episode in ep_indices:
            ep_len = len(self.buffer[episode])
            time_indices.append(np.random.randint(0, ep_len))
        time_indices = np.array(time_indices)

        states = []
        actions = []
        next_states = []
        desired_goals = []
        achieved_goals = []

        # 取出对应回合与时间步的 transitions
        for episode, timestep in zip(ep_indices, time_indices):
            states.append(dc(self.buffer[episode].buffer[timestep][0]))
            actions.append(dc(self.buffer[episode].buffer[timestep][1]))
            next_states.append(dc(self.buffer[episode].buffer[timestep][2]))
            achieved_goals.append(dc(self.buffer[episode].buffer[timestep][4]))
            desired_goals.append(dc(self.buffer[episode].buffer[timestep][5]))

        # 将列表升维并转换成数组类型
        states = np.vstack(states)
        actions = np.vstack(actions)
        desired_goals = np.vstack(desired_goals)
        achieved_goals = np.vstack(achieved_goals)
        next_states = np.vstack(next_states)

        # 根据 future k 概率随机选择要在批量中替换的索引
        her_indices = np.where(np.random.uniform(size=batch_size) < self.future_p)
        future_offset = []
        for episode, timestep in zip(ep_indices, time_indices):
            future_offset.append(np.random.randint(timestep, len(self.buffer[episode])))
        future_offset = np.array(future_offset).astype(int)
        future_t = future_offset[her_indices]

        if task == 'drawer':
            future_ag = []
            for epi, f_offset in zip(ep_indices[her_indices], future_t):
                future_ag.append(dc(self.buffer[epi].buffer[f_offset][4][3:6]))
            future_ag = np.vstack(future_ag)
            desired_goals[her_indices, 3:6] = future_ag
            rewards = np.expand_dims(self.env.compute_reward(achieved_goals[:, 3:6], desired_goals[:, 3:6], th=0.02), 1)
            desired_goals[:, :3] *= 0
            desired_goals[:, 6:] *= 0
        elif task == 'place':
            future_ag = []
            for epi, f_offset in zip(ep_indices[her_indices], future_t):
                future_ag.append(dc(self.buffer[epi].buffer[f_offset][4][3:]))
            future_ag = np.vstack(future_ag)
            desired_goals[her_indices, 3:] = future_ag
            rewards = np.expand_dims(self.env.compute_reward(achieved_goals[:, 3:], desired_goals[:, 3:], th=0.02), 1)
            desired_goals[:, :6] *= 0

        s = torch.tensor(states, dtype=torch.float).to(device)
        a = torch.tensor(actions, dtype=torch.float).to(device)
        s_ = torch.tensor(next_states, dtype=torch.float).to(device)
        r = torch.tensor(rewards, dtype=torch.float).to(device)
        g = torch.tensor(desired_goals, dtype=torch.float).to(device)

        return s, a, s_, r, g

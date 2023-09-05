import copy
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils.replay_buffer import HERReplayBuffer


class Actor(nn.Module):
    def __init__(self, state_dim, goal_dim, action_dim, hidden_dim, max_action):
        super(Actor, self).__init__()
        self.max_action = max_action
        self.fc1 = nn.Linear(state_dim + goal_dim, hidden_dim)  # 输入层
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # 隐藏层
        self.fc3 = nn.Linear(hidden_dim, action_dim)  # 输出层

    def forward(self, s, g):
        s_g = torch.cat([s, g], 1)
        a = F.relu(self.fc1(s_g))
        a = F.relu(self.fc2(a))
        a = self.max_action * torch.tanh(self.fc3(a))
        return a


class Critic(nn.Module):
    def __init__(self, state_dim, goal_dim, action_dim, hidden_dim=128):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim + goal_dim, hidden_dim)  # 输入层
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # 隐藏层
        self.fc3 = nn.Linear(hidden_dim, 1)  # 输出层

    def forward(self, s, g, a):
        s_a = torch.cat([s, g, a], 1)
        q = F.relu(self.fc1(s_a))
        q = F.relu(self.fc2(q))
        return self.fc3(q)


class HERDDPG:
    def __init__(self, env, args):
        self.sigma = args.sigma
        self.agent_name = args.algo_name
        self.device = torch.device(args.device)
        self.gamma = args.gamma  # 奖励的折扣因子
        self.tau = args.tau

        self.batch_size = args.batch_size
        self.memory = HERReplayBuffer(capacity=args.buffer_size, k_future=args.k_future, env=env)

        self.state_dim = args.state_dim
        self.goal_dim = args.goal_dim
        self.action_dim = args.action_dim
        self.hidden_dim = args.hidden_dim
        self.max_action = args.max_action

        self.actor = Actor(self.state_dim, self.goal_dim, self.action_dim, self.hidden_dim, self.max_action).to(
            self.device)
        self.critic = Critic(self.state_dim, self.goal_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.lr)  # 优化器
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.lr)  # 优化器

    def sample_action(self, s, g, deterministic=False):
        with torch.no_grad():
            s = torch.unsqueeze(torch.tensor(s, dtype=torch.float32), 0)
            g = torch.unsqueeze(torch.tensor(g, dtype=torch.float32), 0)
            a = self.actor(s, g).data.numpy().flatten()
            if not deterministic:
                a += self.sigma * np.random.randn(self.action_dim)
                a = np.clip(a, -self.max_action, self.max_action)
                random_actions = np.random.uniform(low=-self.max_action, high=self.max_action,
                                                   size=self.action_dim)
                a += np.random.binomial(1, 0.3, 1)[0] * (random_actions - a)
            return a

    def update(self):
        batch_s, batch_a, batch_s_, batch_r, batch_g = self.memory.sample(self.batch_size)
        q_currents = self.critic(batch_s, batch_g, batch_a)
        with torch.no_grad():  # target_Q has no gradient
            q_next = self.critic_target(batch_s_, batch_g, self.actor_target(batch_s_, batch_g))
            q_targets = batch_r + self.gamma * q_next

        critic_loss = F.mse_loss(q_currents, q_targets)
        self.critic_optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        critic_loss.backward()  # 反向传播更新参数
        self.critic_optimizer.step()

        # Freeze critic networks so you don't waste computational effort
        for params in self.critic.parameters():
            params.requires_grad = False

        actor_loss = -self.critic(batch_s, batch_g, self.actor(batch_s, batch_g)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Unfreeze critic networks
        for params in self.critic.parameters():
            params.requires_grad = True

        # soft update target net
        for params, target_params in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_params.data.copy_(self.tau * params.data + (1 - self.tau) * target_params.data)

        for params, target_params in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_params.data.copy_(self.tau * params.data + (1 - self.tau) * target_params.data)

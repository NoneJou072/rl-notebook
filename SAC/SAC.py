import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from utils.replay_buffer import ReplayBuffer
from utils.ContinuesBase import ContinuesBase
import copy
import numpy as np


class Actor(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.max_action = args.max_action
        self.fc1 = nn.Linear(args.n_states, args.actor_hidden_dim)
        self.fc2 = nn.Linear(args.actor_hidden_dim, args.actor_hidden_dim)
        self.fc3 = nn.Linear(args.actor_hidden_dim, args.n_actions)
        self.log_std = nn.Parameter(
            torch.zeros(1, args.n_actions))  # We use 'nn.Parameter' to train log_std automatically

    def forward(self, s):
        s = F.relu(self.fc1(s))
        s = F.relu(self.fc2(s))
        # 将网络输出的 action 规范在 (-max_action， max_action) 之间
        mean = self.max_action * torch.tanh(self.fc3(s))
        log_std = self.log_std.expand_as(mean)  # To make 'log_std' have the same dimension as 'mean'
        std = torch.exp(log_std)  # The reason we train the 'log_std' is to ensure std=exp(log_std)>0

        dist = Normal(mean, std)  # Get the Gaussian distribution
        a_ = dist.rsample()
        log_pi_ = dist.log_prob(a_).sum(dim=1, keepdim=True)
        # The method refers to Open AI Spinning up, which is more stable.
        log_pi_ -= (2 * (np.log(2) - a_ - F.softplus(-2 * a_))).sum(dim=1, keepdim=True)
        # The method refers to StableBaselines3
        # log_pi_ -= torch.log(1 - a_ ** 2 + self.epsilon).sum(dim=1, keepdim=True)
        a_ = torch.clamp(a_, -self.max_action, self.max_action)

        return a_, log_pi_


class Critic(nn.Module):
    def __init__(self, args):
        super().__init__()
        # Q0
        self.fc1 = nn.Linear(args.n_states, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, 1)
        # Q1
        self.fc4 = nn.Linear(args.n_states, args.hidden_dim)
        self.fc5 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc6 = nn.Linear(args.hidden_dim, 1)

    def forward(self, s, a):
        s_a = torch.cat([s, a], 1)
        # Q1
        q1 = F.relu(self.fc1(s_a))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)
        # Q2
        q2 = F.relu(self.fc1(s_a))
        q2 = F.relu(self.fc2(q2))
        q2 = self.fc3(q2)
        return q1, q2


class SAC(ContinuesBase):
    """
    reference: https://hrl.boyuai.com/chapter/2/sac%E7%AE%97%E6%B3%95/#144-sac
    """
    def __init__(self, args):
        super(SAC, self).__init__(args)
        self.tau = None
        self.alpha = args.alpha
        self.gamma = args.gamma  # discount factor
        self.max_action = args.max_action

        self.batch_size = args.batch_size
        self.buffer_size = args.buffer_size
        self.memory = ReplayBuffer(self.buffer_size)

        self.actor = Actor(args)
        self.critic = Critic(args)
        self.target_critic = copy.deepcopy(self.critic)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr)

    def sample_action(self, s, deterministic=False):
        pass

    def update(self):
        s, a, a_log_prob, s_, r, terminated, done = self.memory.sample(self.batch_size)

        with torch.no_grad():
            # compute target Q value
            a_, log_pi_ = self.actor(s_)
            target_q1, target_q2 = self.target_critic(s_, a_)
            target_q = r + self.gamma * torch.min(target_q1, target_q2) - self.alpha * log_pi_

        # Compute current Q value
        current_q1, current_q2 = self.critic(s, a)
        critic_loss = F.mse_loss(target_q, current_q1) + F.mse_loss(target_q, current_q2)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss
        a, log_pi = self.actor(s)
        q1, q2 = self.critic(s, a)
        actor_loss = self.alpha * log_pi - torch.min(q1, q2).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Softly update target networks
        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

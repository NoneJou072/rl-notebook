import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from utils.replay_buffer import ReplayBuffer
import copy
import numpy as np


class Actor(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.max_action = args.max_action
        self.fc1 = nn.Linear(args.n_states, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, args.n_actions)
        # We use 'nn.Parameter' to train log_std automatically
        self.log_std = nn.Parameter(torch.zeros(1, args.n_actions))

    def forward(self, s, deterministic=False):
        s = F.relu(self.fc1(s))
        s = F.relu(self.fc2(s))
        # 将网络输出的 action 规范在 (-max_action， max_action) 之间
        mean = self.fc3(s)
        log_std = self.log_std.expand_as(mean)  # To make 'log_std' have the same dimension as 'mean'
        std = torch.exp(log_std)  # The reason we train the 'log_std' is to ensure std=exp(log_std)>0

        dist = Normal(mean, std)  # Get the Gaussian distribution
        if deterministic:
            a = mean
        else:
            a = dist.rsample()

        log_pi = dist.log_prob(a).sum(dim=1, keepdim=True)

        # The method refers to Open AI Spinning up, which is more stable.
        log_pi -= (2 * (np.log(2) - a - F.softplus(-2 * a))).sum(dim=1, keepdim=True)
        # The method refers to StableBaselines3
        # log_pi -= torch.log(1 - a ** 2 + self.epsilon).sum(dim=1, keepdim=True)

        # Compress the unbounded Gaussian distribution into a bounded action interval.
        a = torch.clamp(a, -self.max_action, self.max_action)
        # a = self.max_action * torch.tanh(a)

        return a, log_pi


class Critic(nn.Module):
    def __init__(self, args):
        super().__init__()
        # Q0
        self.fc1 = nn.Linear(args.n_states + args.n_actions, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, 1)
        # Q1
        self.fc4 = nn.Linear(args.n_states + args.n_actions, args.hidden_dim)
        self.fc5 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc6 = nn.Linear(args.hidden_dim, 1)

    def forward(self, s, a):
        s_a = torch.cat([s, a], 1)
        # Q1
        q1 = F.relu(self.fc1(s_a))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)
        # Q2
        q2 = F.relu(self.fc4(s_a))
        q2 = F.relu(self.fc5(q2))
        q2 = self.fc6(q2)
        return q1, q2


class SAC:
    """
    reference: https://hrl.boyuai.com/chapter/2/sac%E7%AE%97%E6%B3%95/#144-sac
    """

    def __init__(self, args):

        self.agent_name = 'SAC'

        self.tau = args.tau
        self.lr = args.lr
        self.gamma = args.gamma  # discount factor
        self.max_action = args.max_action
        self.device = args.device
        self.batch_size = args.batch_size
        self.buffer_size = args.buffer_size
        self.memory = ReplayBuffer(self.buffer_size, device=self.device)

        self.target_entropy = -args.n_actions
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], self.lr)

        self.actor = Actor(args).to(self.device)
        self.critic = Critic(args).to(self.device)
        self.target_critic = copy.deepcopy(self.critic)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

    def sample_action(self, s, deterministic=False):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0).to(self.device)
        a, _ = self.actor.forward(s, deterministic)
        return a.cpu().detach().numpy().flatten()

    def update(self):
        s, a, s_, r, terminated, _ = self.memory.sample(self.batch_size, with_log=False)

        with torch.no_grad():
            # compute target Q value
            a_, log_pi_ = self.actor(s_)
            target_q1, target_q2 = self.target_critic(s_, a_)
            target_q = r + self.gamma * (1 - terminated) * torch.min(target_q1, target_q2) - self.alpha * log_pi_

        # Compute current Q value
        current_q1, current_q2 = self.critic(s, a)
        critic_loss = F.mse_loss(target_q, current_q1) + F.mse_loss(target_q, current_q2)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Freeze critic networks so you don't waste computational effort
        for params in self.critic.parameters():
            params.requires_grad = False

        # Compute actor loss
        a, log_pi = self.actor(s)
        q1, q2 = self.critic(s, a)
        actor_loss = (self.alpha * log_pi - torch.min(q1, q2)).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Unfreeze critic networks
        for params in self.critic.parameters():
            params.requires_grad = True

        # Compute temperature loss
        # We learn log_alpha instead of alpha to ensure that alpha=exp(log_alpha)>0
        # https://github.com/rail-berkeley/softlearning/issues/37
        alpha_loss = -(self.log_alpha.exp() * (log_pi + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        # Update alpha
        self.alpha = self.log_alpha.exp()

        # Softly update target networks
        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return actor_loss.item(), critic_loss.item()

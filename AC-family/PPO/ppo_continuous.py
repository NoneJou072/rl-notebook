import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from replay_buffer import PGReplay
import numpy as np


class Actor(nn.Module):
    """ 演员指策略函数，根据当前状态选择回报高的动作。因此 Actor Net 的输入是状态，输出是选择各动作的概率，
    我们希望选择概率尽可能高的动作。 """
    def __init__(self, args) -> None:
        super().__init__()
        self.max_action = args.max_action
        self.fc1 = nn.Linear(args.n_states, args.actor_hidden_dim)
        self.fc2 = nn.Linear(args.actor_hidden_dim, args.actor_hidden_dim)
        self.fc3 = nn.Linear(args.actor_hidden_dim, args.n_actions)
        self.log_std = nn.Parameter(torch.zeros(1, args.n_actions))  # We use 'nn.Parameter' to train log_std automatically
    
    def forward(self, s):
        s = F.relu(self.fc1(s))
        s = F.relu(self.fc2(s))
        # 将网络输出的 action 规范在 (-max_action， max_action) 之间
        mean = self.max_action * torch.tanh(self.fc3(s))
        return mean
    
    def get_dist(self, s):
        mean = self.forward(s)
        log_std = self.log_std.expand_as(mean)  # To make 'log_std' have the same dimension as 'mean'
        std = torch.exp(log_std)  # The reason we train the 'log_std' is to ensure std=exp(log_std)>0
        dist = Normal(mean, std)  # Get the Gaussian distribution
        return dist

class Critic(nn.Module):
    """ 评论员指价值函数，用于对当前策略的值函数进行估计，即评价演员的好坏。因此 Critic Net 的输入是状态，
    输出为在当前策略下该状态上的价值, 维度为1。 """
    def __init__(self, state_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, s):
        s = F.relu(self.fc1(s))
        s = F.relu(self.fc2(s))
        v_s = self.fc3(s)
        return v_s
    

class PPO_continuous:
    def __init__(self, cfg) -> None:

        self.device = torch.device(cfg.device) 
        self.gamma = cfg.gamma 
        self.gae_lambda = cfg.gae_lambda
        self.memory = PGReplay()
        self.k_epochs = cfg.k_epochs # update policy for K epochs
        self.eps_clip = cfg.eps_clip # clip parameter for PPO
        self.entropy_coef = cfg.entropy_coef # entropy coefficient
        self.sample_count = 0
        self.update_freq = cfg.update_freq

        self.actor = Actor(cfg).to(self.device)
        self.critic = Critic(cfg.n_states, hidden_dim=cfg.critic_hidden_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

    @torch.no_grad()
    def sample_action(self,state):
        self.sample_count += 1
        state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
        dist = self.actor.get_dist(state)
        action = dist.sample()
        self.log_probs = dist.log_prob(action).detach().cpu().numpy().flatten()
        return action.detach().cpu().numpy().flatten()
    
    @torch.no_grad()
    def predict_action(self,state):
        state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
        dist = self.actor.get_dist(state)
        action = dist.sample()
        return action.detach().cpu().numpy().flatten()
    
    def evaluate(self, s):  # When evaluating the policy, we only use the mean
        s = torch.unsqueeze(torch.tensor(s, device=self.device, dtype=torch.float), 0)
        a = self.actor(s).detach().cpu().numpy().flatten()
        return a
    
    def update(self):
        # update policy every n steps
        if self.sample_count % self.update_freq != 0:
            return
        # 从replay buffer中采样全部经验, 并转为tensor类型
        old_states, old_actions, old_log_probs, new_states, old_rewards, old_dones = self.memory.sample_tensor(self.device)
        
        with torch.no_grad(): # adv and v_target have no gradient
            # 计算状态价值
            values = self.critic(old_states) 
            new_values = self.critic(new_states)
            # 计算TD误差
            deltas = old_rewards +  self.gamma * values - new_values
            # 计算广义优势
            adv = []
            gae = 0
            for delta, done in zip(reversed(deltas.flatten().cpu().numpy()), reversed(old_dones.flatten().cpu().numpy())):
                gae = delta + self.gamma * self.gae_lambda * (1 - done) * gae
                adv.insert(0, gae)
            # advantage normalization-batch adv norm
            adv = torch.tensor(adv, device=self.device, dtype=torch.float32).view(-1, 1)
            adv = (adv - adv.mean()) / (adv.std() + 1e-5) # 1e-5 to avoid division by zero
        
        for _ in range(self.k_epochs):
            # get action probabilities
            dist = self.actor.get_dist(old_states)
            # get new action probabilities
            new_log_probs = dist.log_prob(old_actions)
            # compute ratio (pi_theta / pi_theta__old):
            ratio = torch.exp(new_log_probs - old_log_probs) # old_log_probs must be detached
            # compute surrogate loss
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * adv
            # compute actor loss
            actor_loss = -torch.min(surr1, surr2).mean() + self.entropy_coef * dist.entropy().mean()
            # compute critic loss
            new_values = self.critic(new_states) 
            critic_loss = F.mse_loss(adv, new_values)
            # take gradient step
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
        self.memory.clear()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from utils.replay_buffer import ReplayBuffer
from utils.ModelBase import ModelBase


class Actor(nn.Module):
    """
    演员指策略函数，根据当前状态选择回报高的动作。因此 Actor Net 的输入是状态，输出是选择各动作的概率，
    我们希望选择概率尽可能高的动作。
    """
    def __init__(self, args) -> None:
        super().__init__()
        self.max_action = args.max_action
        self.fc1 = nn.Linear(args.n_states, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, args.n_actions)
        self.log_std = nn.Parameter(
            torch.zeros(1, args.n_actions))  # We use 'nn.Parameter' to train log_std automatically

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
    """
    评论员指价值函数，用于对当前策略的值函数进行估计，即评价演员的好坏。因此 Critic Net 的输入是状态，
    输出为在当前策略下该状态上的价值, 维度为1。
    """
    def __init__(self, args):
        super().__init__()
        self.fc1 = nn.Linear(args.n_states, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, 1)

    def forward(self, s):
        s = F.relu(self.fc1(s))
        s = F.relu(self.fc2(s))
        v_s = self.fc3(s)
        return v_s


class PPO_continuous(ModelBase):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.device = torch.device(args.device)
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self.k_epochs = args.k_epochs  # update policy for K epochs
        self.eps_clip = args.eps_clip  # clip parameter for PPO
        self.entropy_coef = args.entropy_coef  # entropy coefficient
        self.buffer_size = args.buffer_size
        self.mini_batch_size = args.mini_batch_size
        self.max_action = args.max_action
        self.lr_a = args.actor_lr  # Learning rate of actor
        self.lr_c = args.critic_lr  # Learning rate of critic
        self.max_train_steps = args.max_train_steps

        self.memory = ReplayBuffer(capacity=self.buffer_size)

        self.actor = Actor(args).to(self.device)
        self.critic = Critic(args).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr)

    def sample_action(self, s, deterministic=False):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        if deterministic:
            a = self.actor.forward(s)
            return a.detach().numpy().flatten()
        else:
            with torch.no_grad():
                dist = self.actor.get_dist(s)
                a = dist.sample()
                a = torch.clamp(a, -self.max_action, self.max_action)
                a_logprob = dist.log_prob(a)
            return a.numpy().flatten(), a_logprob.numpy().flatten()

    def update(self, total_steps=0):
        # 从replay buffer中采样全部经验
        old_states, old_actions, old_log_probs, new_states, r, terminated, done = self.memory.sample()
        adv = []
        gae = 0
        with torch.no_grad():  # adv and v_target have no gradient
            # 计算状态价值
            values = self.critic(old_states)
            new_values = self.critic(new_states)
            # 计算TD误差, 注意这里的维度
            deltas = r + self.gamma * (1.0 - terminated) * new_values - values
            # 计算广义优势
            for delta, d in zip(reversed(deltas.flatten().numpy()),
                                reversed(done.flatten().numpy())):
                gae = delta + self.gamma * self.gae_lambda * gae * (1.0 - d)
                adv.insert(0, gae)
            adv = torch.tensor(adv, dtype=torch.float).view(-1, 1)
            v_target = adv + values
            # advantage normalization-batch adv norm
            adv = (adv - adv.mean()) / (adv.std() + 1e-5)  # 1e-5 to avoid division by zero

        for _ in range(self.k_epochs):
            # Random sampling and no repetition. 'False' indicates that training will continue even if the number of
            # samples in the last time is less than mini_batch_size
            for index in BatchSampler(SubsetRandomSampler(range(self.buffer_size)), self.mini_batch_size, False):
                # get action probabilities
                dist = self.actor.get_dist(old_states[index])
                dist_entropy = dist.entropy().sum(1, keepdim=True)  # shape(mini_batch_size X 1)
                # get new action probabilities
                new_log_probs = dist.log_prob(old_actions[index])
                # compute ratio (pi_theta / pi_theta__old), old_log_probs must be detached:
                ratio = torch.exp(new_log_probs.sum(1, keepdim=True) - old_log_probs[index].sum(1, keepdim=True))
                # compute surrogate loss
                surr1 = ratio * adv[index]
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * adv[index]
                # compute actor loss, using policy entropy
                actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy
                self.actor_optimizer.zero_grad()
                actor_loss.mean().backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_optimizer.step()

                # compute critic loss
                v_s = self.critic(old_states[index])
                critic_loss = F.mse_loss(v_target[index], v_s)
                # take gradient step
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optimizer.step()

        self.lr_decay(total_steps)

    def lr_decay(self, total_steps):
        lr_a_now = self.lr_a * (1 - total_steps / self.max_train_steps)
        lr_c_now = self.lr_c * (1 - total_steps / self.max_train_steps)
        for p in self.actor_optimizer.param_groups:
            p['lr'] = lr_a_now
        for p in self.critic_optimizer.param_groups:
            p['lr'] = lr_c_now

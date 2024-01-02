import copy
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from rher_buffer import RHERReplayBuffer


class Actor(nn.Module):
    def __init__(self, state_dim, goal_dim, action_dim, hidden_dim, max_action):
        super(Actor, self).__init__()
        self.max_action = max_action
        self.fc1 = nn.Linear(state_dim + goal_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, action_dim)
        # We use 'nn.Parameter' to train log_std automatically
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))

    def forward(self, s, g, deterministic=False):
        s_g = torch.cat([s, g], 1)
        a = F.relu(self.fc1(s_g))
        a = F.relu(self.fc2(a))
        a = F.relu(self.fc3(a))
        a = F.relu(self.fc4(a))
        # 将网络输出的 action 规范在 (-max_action， max_action) 之间
        mean = self.fc5(a)
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
    def __init__(self, state_dim, goal_dim, action_dim, hidden_dim=128):
        super(Critic, self).__init__()
        # NET1
        self.fc1 = nn.Linear(state_dim + goal_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, 1)
        # NET2
        self.fc6 = nn.Linear(state_dim + goal_dim + action_dim, hidden_dim)
        self.fc7 = nn.Linear(hidden_dim, hidden_dim)
        self.fc8 = nn.Linear(hidden_dim, hidden_dim)
        self.fc9 = nn.Linear(hidden_dim, hidden_dim)
        self.fc10 = nn.Linear(hidden_dim, 1)

    def forward(self, s, g, a):
        s_a = torch.cat([s, g, a], 1)
        q1 = F.relu(self.fc1(s_a))
        q1 = F.relu(self.fc2(q1))
        q1 = F.relu(self.fc3(q1))
        q1 = F.relu(self.fc4(q1))
        q1 = self.fc5(q1)

        q2 = F.relu(self.fc6(s_a))
        q2 = F.relu(self.fc7(q2))
        q2 = F.relu(self.fc8(q2))
        q2 = F.relu(self.fc9(q2))
        q2 = self.fc10(q2)

        return q1, q2


class IORL:
    def __init__(self, env, args):
        self.agent_name = args.algo_name
        self.device = torch.device(args.device)
        self.gamma = args.gamma  # 奖励的折扣因子
        self.tau = args.tau
        self.task_list = args.task_list

        self.training_times = 0

        self.batch_size = args.batch_size
        self.memory_buffer_pool = {}
        for task in self.task_list:
            self.memory_buffer_pool[task] = RHERReplayBuffer(capacity=args.buffer_size, k_future=args.k_future, env=env)

        self.state_dim = args.state_dim
        self.goal_dim = args.goal_dim
        self.action_dim = args.action_dim
        self.hidden_dim = args.hidden_dim
        self.max_action = args.max_action

        self.target_entropy = -self.action_dim
        self.log_alpha = torch.zeros(1).to(self.device)
        self.log_alpha.requires_grad=True
        self.alpha = self.log_alpha.exp().to(self.device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], args.lr)

        self.actor = Actor(self.state_dim, self.goal_dim, self.action_dim, self.hidden_dim, self.max_action).to(self.device)
        self.critic = Critic(self.state_dim, self.goal_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.lr)  # 优化器
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.lr)  # 优化器

        self.critic_loss_record = None
        self.actor_loss_record = None

        self.enable_guide = False

    def check_reached(self, gg, ag, th=0.05):
        """ Check if the gripper has reached the goal position.
        :param gg: goal position
        :param ag: actual position
        :param reward_func: reward function
        :param th: threshold
        :return: True if reached, False otherwise
        """
        grip2obj = np.linalg.norm(gg - ag)

        if grip2obj > th:
            reached = False
        else:
            reached = True
        return reached

    def sample_action(self, s, task, deterministic=False):

        last_obs = copy.deepcopy(s)
        reached = self.check_reached(last_obs['achieved_goal'][:3], last_obs['desired_goal'][:3], th=0.02 if task=='drawer' else 0.02)

        if not deterministic:
            if not reached and self.enable_guide:
                last_obs['desired_goal'][3:] *= 0
            else:
                self.enable_guide = False
                last_obs['desired_goal'][:3] *= 0
                if task == 'drawer':
                    last_obs['desired_goal'][6:] *= 0
                elif task == 'place':
                    last_obs['desired_goal'][3:6] *= 0
        else:
            last_obs['desired_goal'][:3] *= 0
            if task == 'drawer':
                last_obs['desired_goal'][6:] *= 0
            elif task == 'place':
                last_obs['desired_goal'][3:6] *= 0

        with torch.no_grad():
            s = torch.unsqueeze(torch.tensor(last_obs['observation'], dtype=torch.float32), 0).to(self.device)
            g = torch.unsqueeze(torch.tensor(last_obs['desired_goal'], dtype=torch.float32), 0).to(self.device)
            a, _ = self.actor.forward(s, g, deterministic)
            return a.detach().cpu().numpy().flatten()

    def update(self):
        self.training_times += 1
        batch_s_1, batch_a_1, batch_s_1_, batch_r_1, batch_g_1 = self.memory_buffer_pool['reach'].sample(self.batch_size, device=self.device, task='reach')
        batch_s_2, batch_a_1, batch_s_1_, batch_r_1, batch_g_1 = self.memory_buffer_pool['drawer'].sample(self.batch_size, device=self.device, task='drawer')
        batch_s_3, batch_a_1, batch_s_1_, batch_r_1, batch_g_1 = self.memory_buffer_pool['place'].sample(self.batch_size, device=self.device, task='place')

        batch_s = torch.concatenate(
            [self.memory_buffer_pool[task].sample(self.batch_size, device=self.device, task=task)[0] for task in
             self.task_list])
        batch_a = torch.concatenate(
            [self.memory_buffer_pool[task].sample(self.batch_size, device=self.device, task=task)[1] for task in
             self.task_list])
        batch_s_ = torch.concatenate(
            [self.memory_buffer_pool[task].sample(self.batch_size, device=self.device, task=task)[2] for task in
             self.task_list])
        batch_r = torch.concatenate(
            [self.memory_buffer_pool[task].sample(self.batch_size, device=self.device, task=task)[3] for task in
             self.task_list])
        batch_g = torch.concatenate(
            [self.memory_buffer_pool[task].sample(self.batch_size, device=self.device, task=task)[4] for task in
             self.task_list])

        q_currents1, q_currents2 = self.critic(batch_s, batch_g, batch_a)
        with torch.no_grad():  # target_Q has no gradient
            # Clipped dobule Q-learning, compute target Q value
            a_, log_pi_ = self.actor(batch_s_, batch_g)
            q_next1, q_next2 = self.critic_target(batch_s_, batch_g, a_)
            q_targets = batch_r + self.gamma * torch.min(q_next1, q_next2) - self.alpha * log_pi_

        critic_loss = F.mse_loss(q_currents1, q_targets) + F.mse_loss(q_currents2, q_targets)
        self.critic_loss_record = critic_loss.item()
        self.critic_optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        critic_loss.backward()  # 反向传播更新参数
        self.critic_optimizer.step()

        # Delayed policy updates
        for params in self.critic.parameters():
            params.requires_grad = False

        a, log_pi = self.actor(batch_s, batch_g)
        q_currents1, q_currents2 = self.critic(batch_s, batch_g, a)
        actor_loss = (self.alpha * log_pi - torch.min(q_currents1, q_currents2)).mean()
        self.actor_loss_record = actor_loss.item()
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

    def update_target_net(self):
        # soft update target net
        for params, target_params in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_params.data.copy_(self.tau * params.data + (1 - self.tau) * target_params.data)

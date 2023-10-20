import copy
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from rher_buffer import HERReplayBuffer


class Actor(nn.Module):
    def __init__(self, state_dim, goal_dim, action_dim, hidden_dim, max_action):
        super(Actor, self).__init__()
        self.max_action = max_action
        self.fc1 = nn.Linear(state_dim + goal_dim + goal_dim, hidden_dim)  # 输入层
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # 隐藏层
        self.fc3 = nn.Linear(hidden_dim, action_dim)  # 输出层

    def forward(self, s, g, ag):
        s_g = torch.cat([s, g, ag], 1)
        a = F.relu(self.fc1(s_g))
        a = F.relu(self.fc2(a))
        a = self.max_action * torch.tanh(self.fc3(a))
        return a


class Critic(nn.Module):
    def __init__(self, state_dim, goal_dim, action_dim, hidden_dim=128):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + goal_dim + goal_dim + action_dim, hidden_dim)  # 输入层
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # 隐藏层
        self.fc3 = nn.Linear(hidden_dim, 1)  # 输出层

    def forward(self, s, g, ag, a):
        s_a = torch.cat([s, g, ag, a], 1)
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
        self.memory_reach = HERReplayBuffer(capacity=args.buffer_size, k_future=args.k_future, env=env)

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

        self.critic_loss_record = None
        self.actor_loss_record = None

    def check_reached(self, gg, ag, reward_func=None, th=0.03):
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

    def sample_action(self, s, deterministic=False):
        # Here is the part of RHER
        # Trick: SGES 探索策略
        rd = 0.2  # 随机策略选择概率。表示在 reach 阶段， 有 20% 的概率采用随机策略
        rr = 0.4 / (1.0 - rd)  # 0.4 is 自引导率。表示在 reach 阶段，有 40% 的概率采用 reach 模式。

        last_obs = copy.deepcopy(s)
        reached = self.check_reached(last_obs['observation'][:3], last_obs['achieved_goal'])

        if not deterministic:
            # 在 reach 阶段，有 40%的概率采用 reach 策略, 20% 概率采用随机策略，40% 概率采用 push 策略。
            if np.random.random() < rr and not reached:
                reach_a = True  # reach 模式标志位
                last_obs['desired_goal'] *= 0  # Trick：零填充(zero-padding)编码, 用于区分任务
            # 在 push 阶段，有 20%的概率采用随机策略，80% 概率采用 push 策略。
            else:
                reach_a = False
                last_obs['achieved_goal'] *= 0
        else:
            reach_a = False
            last_obs['achieved_goal'] *= 0

        with torch.no_grad():
            s = torch.unsqueeze(torch.tensor(last_obs['observation'], dtype=torch.float32), 0).to(self.device)
            g = torch.unsqueeze(torch.tensor(last_obs['desired_goal'], dtype=torch.float32), 0).to(self.device)
            ag = torch.unsqueeze(torch.tensor(last_obs['achieved_goal'], dtype=torch.float32), 0).to(self.device)
            a = self.actor(s, g, ag).data.cpu().numpy().flatten()
            if not deterministic:
                a += self.sigma * np.random.randn(self.action_dim)  # gaussian noise
                a = np.clip(a, -self.max_action, self.max_action)
                if not reach_a:
                    random_actions = np.random.uniform(low=-self.max_action, high=self.max_action,
                                                       size=self.action_dim)

                    a += np.random.binomial(1, rd, self.action_dim) * (random_actions - a)  # eps-greedy
            return a

    def update(self, rekey='g'):
        if rekey == 'g':
            batch_s, batch_a, batch_s_, batch_r, batch_g, batch_ag = self.memory.sample(self.batch_size,
                                                                                        device=self.device, rekey=rekey)
        else:
            batch_s, batch_a, batch_s_, batch_r, batch_g, batch_ag = self.memory_reach.sample(self.batch_size,
                                                                                        device=self.device, rekey=rekey)

        q_currents = self.critic(batch_s, batch_g, batch_ag, batch_a)
        with torch.no_grad():  # target_Q has no gradient
            q_next = self.critic_target(batch_s_, batch_g, batch_ag, self.actor_target(batch_s_, batch_g, batch_ag))
            q_targets = batch_r + self.gamma * q_next.detach()

        critic_loss = F.mse_loss(q_currents, q_targets)
        self.critic_loss_record = critic_loss.item()
        self.critic_optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        critic_loss.backward()  # 反向传播更新参数
        self.critic_optimizer.step()

        # Freeze critic networks so you don't waste computational effort
        for params in self.critic.parameters():
            params.requires_grad = False

        actor_loss = -self.critic(batch_s, batch_g, batch_ag, self.actor(batch_s, batch_g, batch_ag)).mean()
        self.actor_loss_record = actor_loss.item()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Unfreeze critic networks
        for params in self.critic.parameters():
            params.requires_grad = True

    def update_target_net(self):
        # soft update target net
        for params, target_params in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_params.data.copy_(self.tau * params.data + (1 - self.tau) * target_params.data)

        for params, target_params in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_params.data.copy_(self.tau * params.data + (1 - self.tau) * target_params.data)

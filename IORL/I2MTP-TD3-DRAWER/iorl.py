import copy
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
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

    def forward(self, s, g):
        s_g = torch.cat([s, g], 1)
        a = F.relu(self.fc1(s_g))
        a = F.relu(self.fc2(a))
        a = F.relu(self.fc3(a))
        a = F.relu(self.fc4(a))
        a = self.max_action * torch.tanh(self.fc5(a))
        return a


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
        self.sigma = args.sigma
        self.agent_name = args.algo_name
        self.device = torch.device(args.device)
        self.gamma = args.gamma  # 奖励的折扣因子
        self.tau = args.tau
        self.k_update = args.k_update
        self.training_times = 0

        self.batch_size = args.batch_size
        self.memory_draw = RHERReplayBuffer(capacity=args.buffer_size, k_future=args.k_future, env=env)
        self.memory_place = RHERReplayBuffer(capacity=args.buffer_size, k_future=args.k_future, env=env)
        self.memory_reach = RHERReplayBuffer(capacity=args.buffer_size, k_future=args.k_future, env=env)

        self.state_dim = args.state_dim
        self.goal_dim = args.goal_dim
        self.action_dim = args.action_dim
        self.hidden_dim = args.hidden_dim
        self.max_action = args.max_action

        self.actor = Actor(self.state_dim, self.goal_dim, self.action_dim, self.hidden_dim, self.max_action).to(self.device)
        self.critic = Critic(self.state_dim, self.goal_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
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
            a = self.actor(s, g).data.cpu().numpy().flatten()
            if not deterministic:
                a += self.sigma * np.random.randn(self.action_dim)  # gaussian noise
                a = np.clip(a, -self.max_action, self.max_action)
            return a

    def update(self):
        self.training_times += 1
        batch_s_1, batch_a_1, batch_s_1_, batch_r_1, batch_g_1 = self.memory_reach.sample(self.batch_size,
                                                                                          device=self.device,
                                                                                          task='reach')
        batch_s_2, batch_a_2, batch_s_2_, batch_r_2, batch_g_2 = self.memory_draw.sample(self.batch_size,
                                                                                         device=self.device,
                                                                                         task='drawer')
        batch_s_3, batch_a_3, batch_s_3_, batch_r_3, batch_g_3 = self.memory_place.sample(self.batch_size,
                                                                                         device=self.device,
                                                                                         task='place')

        batch_s = torch.concatenate((batch_s_1, batch_s_2, batch_s_3))
        batch_a = torch.concatenate((batch_a_1, batch_a_2, batch_a_3))
        batch_s_ = torch.concatenate((batch_s_1_, batch_s_2_, batch_s_3_))
        batch_r = torch.concatenate((batch_r_1, batch_r_2, batch_r_3))
        batch_g = torch.concatenate((batch_g_1, batch_g_2, batch_g_3))

        q_currents1, q_currents2 = self.critic(batch_s, batch_g, batch_a)
        with torch.no_grad():  # target_Q has no gradient
            # Clipped dobule Q-learning
            q_next1, q_next2 = self.critic_target(batch_s_, batch_g, self.actor_target(batch_s_, batch_g))
            q_targets = batch_r + self.gamma * torch.min(q_next1, q_next2)

        critic_loss = F.mse_loss(q_currents1, q_targets) + F.mse_loss(q_currents2, q_targets)
        self.critic_loss_record = critic_loss.item()
        self.critic_optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        critic_loss.backward()  # 反向传播更新参数
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.training_times % self.k_update == 0:
            # Freeze critic networks so you don't waste computational effort
            for params in self.critic.parameters():
                params.requires_grad = False

            q_currents1, q_currents2 = self.critic(batch_s, batch_g, self.actor(batch_s, batch_g))
            actor_loss = -torch.min(q_currents1, q_currents2).mean()
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

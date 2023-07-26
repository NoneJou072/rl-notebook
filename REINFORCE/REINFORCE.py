import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)  # 输入层
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # 隐藏层
        self.fc3 = nn.Linear(hidden_dim, action_dim)  # 输出层

    def forward(self, s):
        s = F.relu(self.fc1(s))
        s = F.relu(self.fc2(s))
        a_prob = F.softmax(self.fc3(s), dim=1)
        return a_prob
    

class REINFORCE:
    def __init__(self, args):
        self.agent_name = args.algo_name
        self.lr = args.lr
        self.gamma = args.gamma
        self.state_dim = args.state_dim
        self.action_dim = args.action_dim
        self.hidden_dim = args.action_dim

        self.states = []
        self.actions = []
        self.rewards = []

        self.actor = Actor(self.state_dim, self.action_dim, self.hidden_dim)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)

    @torch.no_grad()
    def sample_action(self, s, deterministic=False):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float32), 0)
        a_prob = self.actor(s).data.numpy().flatten()
        if deterministic:
            # Select the action with the highest probability
            a = np.argmax(a_prob)
        else:
            # Sample the action according to the probability distribution
            a = np.random.choice(range(self.action_dim), p=a_prob)
        return a

    def update(self, s, a, r, d):
        """ 策略更新函数，这里实现伪代码中的更新公式 """
        # Append transition to the buffer
        self.states.append(s)
        self.actions.append(a)
        self.rewards.append(r)

        if d:
            G = []
            g = 0
            # 计算当前轨迹每个时刻往后的回报 g
            for r in reversed(self.rewards):
                g = self.gamma * g + r
                G.insert(0, g)

            # 更新策略
            for t in range(len(self.rewards)):
                s = torch.unsqueeze(torch.tensor(self.states[t], dtype=torch.float32), 0)
                a = self.actions[t]
                g = G[t]

                a_prob = self.actor(s).flatten()
                loss = -self.gamma ** t * g * torch.log(a_prob[a])
                self.actor_optimizer.zero_grad()
                loss.backward()
                self.actor_optimizer.step()

            # Clean the buffers
            self.states = []
            self.actions = []
            self.rewards = []
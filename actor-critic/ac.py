import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, s):
        s = F.relu(self.fc1(s))
        a_prob = F.softmax(self.fc2(s), dim=1)
        return a_prob
    

class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim) -> None:
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, s):
        v = F.relu(self.fc1(s))
        return self.fc2(v)
    

class AC:
    def __init__(self, args):
        self.agent_name = args.algo_name
        self.lr = args.lr
        self.gamma = args.gamma
        self.tau = args.tau

        self.state_dim = args.state_dim
        self.action_dim = args.action_dim
        self.hidden_dim = args.hidden_dim

        self.actor = Actor(self.state_dim, self.action_dim, self.hidden_dim)
        self.critic = Critic(self.state_dim, self.hidden_dim)
    
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)
    
    @torch.no_grad()
    def sample_action(self, s, deterministic=False):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float32), 0)
        a_prob = self.actor(s)
        if deterministic:
            # Select the action with the highest probability
            a = np.argmax(a_prob.numpy().flatten())
        else:
            # Sample the action according to the probability distribution
            # a = np.random.choice(range(self.action_dim), p=a_prob)
            action_dist = torch.distributions.Categorical(a_prob)
            a = action_dist.sample().item()
        return a

    def update(self, s, a, r, s_, d):
        """ 策略更新函数，这里实现伪代码中的更新公式 """
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float32), 0)
        s_ = torch.unsqueeze(torch.tensor(s_, dtype=torch.float32), 0)

        v = self.critic(s).flatten()

        with torch.no_grad():
            v_ = self.critic(s_).flatten()
            td_target = r + self.gamma * v_ * (1 - d)

        log_pi = torch.log(self.actor(s).flatten()[a])  # log pi(a|s)
        actor_loss = -(td_target.detach()) * log_pi
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        critic_loss = F.mse_loss(v, td_target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()


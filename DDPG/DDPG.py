import copy
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils.replay_buffer import ReplayBuffer


class QNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        """
        初始化Q网络为全连接网络
        """
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)  # 输入层
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # 隐藏层
        self.fc3 = nn.Linear(hidden_dim, action_dim)  # 输出层

    def forward(self, x):
        # 各层对应的激活函数
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DQN:
    """
    use_soft_update: hard update 与 soft update 的区别参考
        https://mp.weixin.qq.com/s/p9B8f8nA-nclrRe9ypHarw
    """
    def __init__(self, args):
        self.agent_name = args.algo_name
        self.device = torch.device(args.device)
        self.gamma = args.gamma  # 奖励的折扣因子
        self.epsilon = args.epsilon
        # self.epsilon_start = args['epsilon_start']
        # self.epsilon_end = args['epsilon_end']
        # self.epsilon_decay = args['epsilon_decay']
        
        self.update_count = 0
        self.update_frequence = args.update_frequence

        self.batch_size = args.batch_size
        self.memory = ReplayBuffer(capacity=args.buffer_size)

        self.state_dim = args.state_dim
        self.action_dim = args.action_dim
        self.hidden_dim = args.hidden_dim
        self.policy_net = QNet(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.target_net = copy.deepcopy(self.policy_net)

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=args.lr)  # 优化器

    def sample_action(self, s, deterministic=False):
        with torch.no_grad():
            s = torch.unsqueeze(torch.tensor(s, dtype=torch.float32), 0) 
            if np.random.uniform() > self.epsilon or deterministic:
                action = self.policy_net(s).argmax(dim=-1).item()
            else:
                action = np.random.randint(0, self.action_dim)
            return action

    def update(self):
        batch_s, batch_a, batch_s_, batch_r, batch_terminated, batch_d = self.memory.sample(self.batch_size, with_log=False)
        q_currents = self.policy_net(batch_s)
        q_currents = q_currents.gather(-1, batch_a.unsqueeze(-1))  # shape：(batch_size,)

        max_q_target = self.target_net(batch_s_).max(dim=-1)[0]
        q_targets = batch_r + self.gamma * max_q_target.unsqueeze(-1) * (1 - batch_terminated)
        td_errors = q_currents - q_targets
        loss = (td_errors ** 2).mean()

        self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        loss.backward()  # 反向传播更新参数
        self.optimizer.step()
        
        # hard update
        self.update_count += 1
        if self.update_count % self.update_frequence == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

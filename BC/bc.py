import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        """
        初始化Q网络为全连接网络
        """
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)  # 输入层
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # 隐藏层
        self.fc3 = nn.Linear(hidden_dim, action_dim)  # 输出层

    def forward(self, x):
        # 各层对应的激活函数
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    

class BehaviorClone:
    def __init__(self, state_dim, action_dim, hidden_dim, lr):
        self.policy = MLP(state_dim, action_dim, hidden_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def learn(self, states, actions):
        states = torch.tensor(states, dtype=torch.float).to(device)
        actions = torch.tensor(actions).to(device)
        actions_ = self.policy(states)

        bc_loss = F.mse_loss(actions_, actions)

        self.optimizer.zero_grad()
        bc_loss.backward()
        self.optimizer.step()

        return bc_loss.item()

    def take_action(self, state: np.ndarray):
        state = torch.tensor(np.expand_dims(state, axis=0), dtype=torch.float).to(device)
        action = self.policy(state)
        return action.flatten().cpu().detach().numpy()

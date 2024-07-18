import torch
import torch.nn as nn
import numpy as np


class Discriminator(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        self.fc3.weight.data.mul_(0.1)
        self.fc3.bias.data.mul_(0.0)

    def forward(self, s, a):
        x = torch.cat([s, a], dim=1)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        prob = torch.sigmoid(self.fc3(x))
        return prob


class GAIL:
    def __init__(self, args):
        self.state_dim = args.n_states
        self.action_dim = args.n_actions
        self.hidden_dim = args.hidden_dim
        self.lr = args.lr
        self.device = args.device

        self.discriminator = Discriminator(
            self.state_dim, self.action_dim, 128
        ).to(self.device)
        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.lr
        )

    def learn(
            self, 
            expert_s, 
            expert_a, 
            memory_buffer
        ):
        expert_states = torch.tensor(expert_s, dtype=torch.float).to(self.device)
        expert_actions = torch.tensor(expert_a).to(self.device)

        if len(memory_buffer.buffer[0]) == 6: 
            agent_states, agent_actions, _, _, _, _ = memory_buffer.sample(with_log=False)
        else:
            agent_states, agent_actions, _, _, _, _, _ = memory_buffer.sample()
        
        criterion = torch.nn.BCELoss()

        for _ in range(2):
            # 给专家的动作-状态对打高分，给模仿者的动作-状态对打低分，分数是用概率值体现的
            expert_prob = self.discriminator(expert_states, expert_actions)
            agent_prob = self.discriminator(agent_states, agent_actions)

            # 希望生成器生成的数据被判别器误认为是真实数据，从而迷惑判别器，所以判别器的目标是尽量将专家数据的输出靠近 0，将模仿者策略的输出靠近 1，并将两组数据分辨开来。
            discrim_loss = criterion(agent_prob, torch.ones_like(agent_prob)) + \
                                criterion(expert_prob, torch.zeros_like(expert_prob))
            
            self.discriminator_optimizer.zero_grad()
            discrim_loss.backward()
            self.discriminator_optimizer.step()

        expert_prob = self.discriminator(expert_states, expert_actions)
        agent_prob = self.discriminator(agent_states, agent_actions)

        expert_acc = ((expert_prob < 0.5).float()).mean()
        learner_acc = ((agent_prob > 0.5).float()).mean()

        return discrim_loss.item(), expert_acc, learner_acc

    def get_reward(self, state: np.ndarray, action: np.ndarray):
        state = torch.tensor(np.expand_dims(state, axis=0), dtype=torch.float32).to(self.device)
        action = torch.tensor(np.expand_dims(action, axis=0), dtype=torch.float32).to(self.device)
        # agent_prob 越接近 0，说明判别器认为这个动作-状态对是专家的
        # 因此我们需要对这个概率取负对数，作为奖励值
        agent_prob = self.discriminator(state, action)
        with torch.no_grad():
            return -torch.log(agent_prob + 1e-8).detach().cpu().numpy().item()
    
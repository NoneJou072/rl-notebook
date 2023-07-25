import numpy as np
from collections import defaultdict

class Sarsa:
    def __init__(self, args):
        self.epsilon = None
        self.agent_name = args.algo_name
        self.lr = args.lr
        self.gamma = args.gamma
        self.action_dim = args.action_dim
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_start = args.epsilon_start
        self.epsilon_end = args.epsilon_end
        self.sample_count = 0
        # 用嵌套字典存放状态->动作->状态-动作值（Q值）的映射
        self.Q_table = defaultdict(lambda: np.zeros(self.action_dim))
        
    def sample_action(self, state, deterministic=False):
        self.sample_count += 1
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                       np.exp(-1. * self.sample_count / self.epsilon_decay)  # epsilon是会递减的，这里选择指数递减
        # 带有探索的贪心策略
        if np.random.uniform(0, 1) > self.epsilon or deterministic:
            action = np.argmax(self.Q_table[str(state)])  # 选择Q(s,a)最大值对应的动作
        else:
            action = np.random.choice(self.action_dim)  # 随机选择动作
        return action

    def update(self, state, action, reward, next_state, next_action, done):
        """ 策略更新函数，这里实现伪代码中的更新公式 """
        Q_predict = self.Q_table[str(state)][action]
        Q_target = reward + self.gamma * (1 - done) * self.Q_table[str(next_state)][next_action]
        # 更新 Q 表
        self.Q_table[str(state)][action] += self.lr * (Q_target - Q_predict)

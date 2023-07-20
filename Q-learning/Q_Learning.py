class QLearning:
    def __init__(self):
        self.agent = QLearning(n_states, n_actions, cfg)  # cfg存储算法相关参数

    def choose_action(self, state):
        self.sample_count += 1
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                       math.exp(-1. * self.sample_count / self.epsilon_decay)  # epsilon是会递减的，这里选择指数递减
        # 带有探索的贪心策略
        if np.random.uniform(0, 1) > self.epsilon:
            action = np.argmax(self.Q_table[str(state)])  # 选择Q(s,a)最大值对应的动作
        else:
            action = np.random.choice(self.action_dim)  # 随机选择动作
        return action

    def update(self, state, action, reward, next_state, done):
        Q_predict = self.Q_table[str(state)][action]
        if done:  # 终止状态
            Q_target = reward
        else:
            Q_target = reward + self.gamma * np.max(self.Q_table[str(next_state)])
        self.Q_table[str(state)][action] += self.lr * (Q_target - Q_predict)

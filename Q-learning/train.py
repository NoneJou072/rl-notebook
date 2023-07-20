import os
import numpy as np
from Q_Learning import QLearning
from utils.ModelBase import ModelBase
import argparse
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym


class QLearningModel(ModelBase):
    def __init__(self, env, args):
        super().__init__(env, args)
        self.agent = DQN(args)
        self.model_name = f'{self.agent.agent_name}_{self.args.env_name}_num_{1}_seed_{self.args.seed}'

    def train(self):
        rewards = []
        ma_rewards = []  # 滑动平均奖励
        for i_ep in range(cfg.train_eps):
            ep_reward = 0  # 记录每个回合的奖励
            state = env.reset()  # 重置环境, 重新开始（开始一个新的回合）
            while True:
                action = agent.choose_action(state)  # 根据算法选择一个动作
                next_state, reward, done, _ = env.step(action)  # 与环境进行一次动作交互
                agent.update(state, action, reward, next_state, done)  # Q学习算法更新
                state = next_state  # 存储上一个观察值
                ep_reward += reward
                if done:
                    break
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(ma_rewards[-1] * 0.9 + ep_reward * 0.1)
        else:
            ma_rewards.append(ep_reward)


def make_env(args):
    """ 配置智能体和环境 """
    env = gym.make('CliffWalking-v0')  # 定义环境

    state_dim = env.observation_space.n  # 状态数
    action_dim = env.action_space.n  # 动作数
    print(f"state dim:{state_dim}, action dim:{action_dim}")
    setattr(args, 'state_dim', state_dim)
    setattr(args, 'action_dim', action_dim)
    setattr(args, 'max_episode_steps', env._max_episode_steps)

    return env


if __name__ == '__main__':
    args = Config().__call__()
    env = make_env(args)
    model = QLearningModel(
        env=env,
        args=args,
    )
    model.train()

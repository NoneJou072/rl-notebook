import os
import numpy as np
import torch
from utils.ModelBase import ModelBase

from HER.HERDDPG import HERDDPG
import argparse
import gymnasium as gym

local_path = os.path.dirname(__file__)
log_path = os.path.join(local_path, 'log')


def args():
    parser = argparse.ArgumentParser("Hyperparameters Setting for DDPG")
    parser.add_argument("--env_name", type=str, default="FetchPickAndPlace-v2", help="env name")
    parser.add_argument("--algo_name", type=str, default="HERDDPG", help="algorithm name")
    parser.add_argument("--seed", type=int, default=10, help="random seed")
    parser.add_argument("--device", type=str, default='cpu', help="pytorch device")
    # Training Params
    parser.add_argument("--max_train_steps", type=int, default=int(3e5), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=5e3,
                        help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=10, help="Save frequency")
    parser.add_argument("--buffer_size", type=int, default=int(1e6), help="Reply buffer size")
    parser.add_argument("--batch_size", type=int, default=256, help="Minibatch size")
    # Net Params
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate of QNet")
    parser.add_argument("--gamma", type=float, default=0.98, help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.05, help="Softly update the target network")
    parser.add_argument("--use_state_norm", type=bool, default=False, help="Trick: state normalization")
    parser.add_argument("--random_steps", type=int, default=1e3,
                        help="Take the random actions in the beginning for the better exploration")
    parser.add_argument("--update_freq", type=int, default=50, help="Take 50 steps,then update the networks 50 times")
    parser.add_argument("--k_future", type=int, default=4, help="Her k future")

    return parser.parse_args()


class HERDDPGModel(ModelBase):
    def __init__(self, env, args):
        super().__init__(env, args)
        self.agent = HERDDPG(env, args)
        self.model_name = f'{self.agent.agent_name}_{self.args.env_name}_num_{1}_seed_{self.args.seed}'
        self.load_weights()

    def load_weights(self):
        model_dir = os.path.join(log_path, f'./data_train/{self.agent.agent_name}')
        model_path = os.path.join(model_dir, f'{self.model_name}.pth')
        actor_state_dict = torch.load(model_path)
        self.agent.actor.load_state_dict(actor_state_dict)

    def play(self):
        total_steps = 0
        while total_steps < self.args.max_train_steps:
            env_dict, _ = self.env.reset()  # 重置环境，返回初始状态
            s = env_dict["observation"]
            achieved_g = env_dict["achieved_goal"]
            desired_g = env_dict["desired_goal"]
            while np.linalg.norm(achieved_g - desired_g) <= 0.05:
                env_dict, _ = self.env.reset()
                s = env_dict["observation"]
                achieved_g = env_dict["achieved_goal"]
                desired_g = env_dict["desired_goal"]
            while True:
                total_steps += 1
                a = self.agent.sample_action(s, desired_g, deterministic=True)  # 选择动作
                print(a)
                env_dict_, r, terminated, truncated, _ = self.env.step(a)  # 更新环境，返回transition
                s = env_dict_["observation"].copy()
                desired_g = env_dict_["desired_goal"].copy()
                if truncated:
                    break

        self.env.close()


def make_env(args):
    """ 配置环境 """
    env = gym.make(args.env_name, render_mode='human')  # 创建环境
    state_dim = env.observation_space.spaces["observation"].shape[0]
    action_dim = env.action_space.shape[0]
    goal_dim = env.observation_space.spaces["desired_goal"].shape[0]
    max_action = float(env.action_space.high[0])
    min_action = float(env.action_space.low[0])

    setattr(args, 'state_dim', state_dim)
    setattr(args, 'action_dim', action_dim)
    setattr(args, 'goal_dim', goal_dim)
    setattr(args, 'max_episode_steps', env._max_episode_steps)
    setattr(args, 'max_action', max_action)
    setattr(args, 'sigma', 0.2)  # The std of Gaussian noise for exploration

    return env


if __name__ == '__main__':
    args = args()
    env = make_env(args)
    model = HERDDPGModel(
        env=env,
        args=args,
    )
    model.play()

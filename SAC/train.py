import os
import numpy as np
from SAC import SAC
from utils.ModelBase import ModelBase
import argparse
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
from utils.normalization import Normalization

local_path = os.path.dirname(__file__)
log_path = os.path.join(local_path, 'log')


class Config:
    def __call__(self, *args, **kwargs):
        # Env Params
        parser = argparse.ArgumentParser("Hyperparameters Setting for SAC")
        parser.add_argument("--env_name", type=str, default="Walker2d-v2", help="env name")
        parser.add_argument("--algo_name", type=str, default="SAC", help="algorithm name")
        parser.add_argument("--seed", type=int, default=10, help="random seed")
        parser.add_argument("--device", type=str, default='cpu', help="pytorch device")
        # Training Params
        parser.add_argument("--max_train_steps", type=int, default=int(3e6), help=" Maximum number of training steps")
        parser.add_argument("--max_episode_steps", type=int, default=int(1e3), help=" Maximum number of training steps")
        parser.add_argument("--evaluate_freq", type=float, default=5e3,
                            help="Evaluate the policy every 'evaluate_freq' steps")
        parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
        parser.add_argument("--buffer_size", type=int, default=int(1e6), help="Reply buffer size")
        parser.add_argument("--batch_size", type=int, default=256, help="Minibatch size")
        # Net Params
        parser.add_argument("--hidden_dim", type=int, default=256,
                            help="The number of neurons in hidden layers of the neural network")
        parser.add_argument("--actor_lr", type=float, default=3e-4, help="Learning rate of actor")
        parser.add_argument("--critic_lr", type=float, default=3e-4, help="Learning rate of critic")
        parser.add_argument("--alpha_lr", type=float, default=3e-4, help="Learning rate of alpha")
        parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
        parser.add_argument("--tau", type=float, default=0.005, help="Softly update the target network")
        parser.add_argument("--k_epochs", type=int, default=10, help="更新策略网络的次数")
        parser.add_argument("--use_state_norm", type=bool, default=False, help="Trick 2:state normalization")

        return parser.parse_args()


class SACModel(ModelBase):
    def __init__(self, env, args):
        super().__init__(env, args)
        self.agent = SAC(args)
        self.model_name = f'{self.agent.agent_name}_{self.args.env_name}_num_{1}_seed_{self.args.seed}'
        if self.args.use_state_norm:
            self.state_norm = Normalization(shape=self.args.n_states)  # Trick 2:state normalization

    def train(self):
        """ 训练 """
        print("开始训练！")

        total_steps = 0
        evaluate_num = 0
        sample_count = 0
        evaluate_rewards = []  # 记录每回合的奖励

        # Tensorboard config
        log_dir = os.path.join(log_path, f'./runs/{self.model_name}')
        writer = SummaryWriter(log_dir=log_dir)

        while total_steps < self.args.max_train_steps:
            s, _ = self.env.reset(seed=self.args.seed)  # 重置环境，返回初始状态
            ep_step = 0
            while True:
                ep_step += 1
                sample_count += 1
                a = self.agent.sample_action(s)  # 选择动作
                s_, r, terminated, truncated, _ = self.env.step(a)  # 更新环境，返回transition

                if ep_step == self.args.max_episode_steps:
                    truncated = True

                # 保存transition
                self.agent.memory.push((s, a, s_, r, terminated, truncated))
                s = s_  # 更新下一个状态
                total_steps += 1

                # update policy every steps
                self.agent.update()

                if total_steps % self.args.evaluate_freq == 0:
                    evaluate_num += 1
                    evaluate_reward = self.evaluate_policy()
                    evaluate_rewards.append(evaluate_reward)
                    print("evaluate_num:{} \t evaluate_reward:{} \t".format(evaluate_num, evaluate_reward))
                    writer.add_scalar('step_rewards_{}'.format(self.args.env_name), evaluate_rewards[-1],
                                      global_step=total_steps)
                    # Save the rewards
                    if evaluate_num % self.args.save_freq == 0:
                        model_dir = os.path.join(log_path, f'./data_train/{self.agent.agent_name}')
                        if not os.path.exists(model_dir):
                            os.makedirs(model_dir)
                        np.save(os.path.join(model_dir, f'{self.model_name}.npy'), np.array(evaluate_rewards))
                if terminated or truncated:
                    break

        print("完成训练！")
        self.env.close()


def make_env(args):
    """ 配置智能体和环境 """
    env = gym.make(args.env_name)  # 创建环境
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]
    print(f"state dim:{state_dim}, action dim:{action_dim}, max action:{max_action}")

    # 更新n_states, max_action和n_actions到cfg参数中
    setattr(args, 'state_dim', state_dim)
    setattr(args, 'action_dim', action_dim)
    setattr(args, 'max_action', max_action)

    return env


if __name__ == '__main__':
    args = Config().__call__()
    env = make_env(args)
    model = SACModel(
        env=env,
        args=args,
    )
    model.train()

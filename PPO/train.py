import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ppo_continuous import PPO_continuous
from utils.ModelBase import ModelBase
from utils.normalization import Normalization
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
from utils.normalization import Normalization
import gymnasium as gym

local_path = os.path.dirname(__file__)
log_path = os.path.join(local_path, 'log')

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Env Params
parser = argparse.ArgumentParser("Hyperparameters Setting for PPO-continuous")
parser.add_argument("--env_name", type=str, default="Walker2d-v4", help="env name")
parser.add_argument("--algo_name", type=str, default="PPO-continuous", help="algorithm name")
parser.add_argument("--seed", type=int, default=10, help="random seed")
parser.add_argument("--device", type=str, default=device, help="pytorch device")
# Training Params
parser.add_argument("--max_train_steps", type=int, default=int(1e6), help=" Maximum number of training steps")
parser.add_argument("--max_episode_horizon", type=int, default=512, help=" Maximum number of training steps")
parser.add_argument("--buffer_size", type=int, default=2048, help="Reply buffer size")
parser.add_argument("--mini_batch_size", type=int, default=64, help="Minibatch size")
# Net Params
parser.add_argument("--hidden_dim", type=int, default=64,
                    help="The number of neurons in hidden layers of the neural network")
parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE parameter")
parser.add_argument("--eps_clip", type=float, default=0.2, help="PPO clip parameter-epsilon-clip")
parser.add_argument("--k_epochs", type=int, default=10, help="PPO parameter, 更新策略网络的次数")
parser.add_argument("--entropy_coef", type=float, default=0.01, help="policy entropy")
parser.add_argument("--max_grad_norm", type=float, default=0.5, help="max_grad_norm")

# Optim tricks， reference by https://zhuanlan.zhihu.com/p/512327050
parser.add_argument("--use_state_norm", type=bool, default=True, help="Trick 2:state normalization")
parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
args = parser.parse_args()


class PPOContinuousModel(ModelBase):
    def __init__(self, env, args):
        super().__init__(env, args)
        self.agent = PPO_continuous(args)
        self.model_name = f'PPO_{self.args.env_name}_num_{1}_seed_{self.args.seed}'
        if self.args.use_state_norm:
            self.state_norm = Normalization(shape=self.args.n_states)  # Trick 2:state normalization

    def train(self):
        """ 训练 """

        total_steps = 0
        evaluate_num = 0

        # Tensorboard config
        log_dir = os.path.join(log_path, f'./runs/{self.model_name}')
        writer = SummaryWriter(log_dir=log_dir)
        print("开始训练！")

        while total_steps < self.args.max_train_steps:
            s, _ = self.env.reset(seed=self.args.seed)  # 重置环境，返回初始状态
            if self.args.use_state_norm:
                s = self.state_norm(s)
            ep_step = 0
            while True:
                a, a_logprob = self.agent.sample_action(s)  # 选择动作
                s_, r, terminated, truncated, _ = self.env.step(a)  # 更新环境，返回transition

                if self.args.use_state_norm:
                    s_ = self.state_norm(s_)

                if ep_step == self.args.max_episode_horizon:
                    truncated = True

                # 保存transition
                self.agent.memory.push((s, a, a_logprob, s_, r, terminated, truncated))
                s = s_  # 更新下一个状态
                total_steps += 1
                ep_step += 1

                # update policy every n steps
                if total_steps % self.args.buffer_size == 0:
                    actor_loss, critic_loss = self.agent.update(total_steps)
                    self.agent.memory.clear()

                    evaluate_num += 1
                    ep_reward, ep_len = self.evaluate_policy()

                    print("total_steps:{} \t ep_reward:{} \t ep_len:{} \t".format(total_steps, ep_reward, ep_len))
                    print(f"actor_loss:{actor_loss}, critic_loss:{critic_loss}")
                    writer.add_scalar('step_reward_{}'.format(self.args.env_name), ep_reward,
                                        global_step=total_steps)
                    writer.add_scalar('step_len_{}'.format(self.args.env_name), ep_len,
                                        global_step=total_steps)
                    # Save the model
                    model_dir = os.path.join(log_path, f'./data_train/{self.model_name}.npy')
                    pass

                if terminated or truncated:
                    break

        print("完成训练！")
        self.env.close()


def make_env(args):
    """ 配置智能体和环境 """
    env = gym.make(args.env_name)  # 创建环境
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    max_action = env.action_space.high[0]
    print(f"state dim:{n_states}, action dim:{n_actions}, max action:{max_action}")

    # 更新n_states, max_action和n_actions到cfg参数中
    setattr(args, 'n_states', n_states)
    setattr(args, 'n_actions', n_actions)
    setattr(args, 'max_action', max_action)

    return env


if __name__ == '__main__':

    env = make_env(args)
    model = PPOContinuousModel(
        env=env,
        args=args,
    )
    model.train()

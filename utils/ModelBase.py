import os
import gymnasium as gym
import numpy as np
import torch
import argparse
from torch.utils.tensorboard import SummaryWriter
from utils.normalization import Normalization

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print('CUDA version:', torch.version.cuda)  # 打印CUDA版本
print('CUDA available:', torch.cuda.is_available())  # 检查CUDA是否可用

local_path = os.path.dirname(__file__)
log_path = os.path.join(local_path, 'log')


class ModelBase(object):
    def __init__(self, env: gym.Env, args: argparse.Namespace):
        self.args = args
        self.env = env
        self.env_evaluate = env
        self.set_seed(args.seed)

        self.agent = None
        self.model_name = None
        self.state_norm = None

    def evaluate_policy(self):
        times = 5
        evaluate_reward = 0
        evaluate_episode_len = 0
        for _ in range(times):
            s, _ = self.env_evaluate.reset(seed=self.args.seed)
            if self.args.use_state_norm:
                s = self.state_norm(s, update=False)  # During the evaluating,update=False
            episode_reward = 0
            episode_len = 0
            while True:
                # We use the deterministic policy during the evaluating
                action = self.agent.sample_action(s, deterministic=True)
                s_, r, terminated, truncated, _ = self.env_evaluate.step(action)
                if self.args.use_state_norm:
                    s_ = self.state_norm(s_, update=False)
                s = s_
                episode_reward += r
                episode_len += 1
                if terminated or truncated:
                    break
            evaluate_reward += episode_reward
            evaluate_episode_len += episode_len

        return evaluate_reward / times, evaluate_episode_len / times

    def set_seed(self, seed=10):
        """ 配置seed """
        if seed == 0:
            return
        self.env.action_space.seed(seed)
        self.env_evaluate.action_space.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)  # config for CPU
        torch.cuda.manual_seed(seed)  # config for GPU
        os.environ['PYTHONHASHSEED'] = str(seed)  # config for python scripts

    def train(self):
        """ 训练 """
        print("开始训练！")

        total_steps = 0
        evaluate_num = 0
        sample_count = 0

        # Tensorboard config
        log_dir = os.path.join(log_path, f'./runs/{self.model_name}')
        writer = SummaryWriter(log_dir=log_dir)

        while total_steps < self.args.max_train_steps:
            s, _ = self.env.reset(seed=self.args.seed)  # 重置环境，返回初始状态
            ep_step = 0
            while True:
                ep_step += 1
                sample_count += 1
                a, a_logprob = self.agent.sample_action(s)  # 选择动作
                s_, r, terminated, truncated, _ = self.env.step(a)  # 更新环境，返回transition

                if ep_step == self.args.max_episode_steps:
                    truncated = True

                # 保存transition
                self.agent.memory.push((s, a, a_logprob, s_, r, terminated, truncated))
                s = s_  # 更新下一个状态
                total_steps += 1

                # update policy every n steps
                if sample_count % self.args.buffer_size == 0:
                    self.agent.update(total_steps)

                if total_steps % self.args.evaluate_freq == 0:
                    evaluate_num += 1
                    ep_reward, ep_len = self.evaluate_policy()
                    print("total_steps:{} \t ep_reward:{} \t ep_len:{} \t".format(total_steps, ep_reward, ep_len))
                    writer.add_scalar('step_reward_{}'.format(self.args.env_name), ep_reward,
                                      global_step=total_steps)
                    writer.add_scalar('step_len_{}'.format(self.args.env_name), ep_len,
                                      global_step=total_steps)
                    # Save the model
                    if evaluate_num % self.args.save_freq == 0:
                        model_dir = os.path.join(log_path, f'./data_train/{self.model_name}.npy')
                        pass
                if terminated or truncated:
                    break

        print("完成训练！")
        self.env.close()

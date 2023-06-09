import os
import gymnasium as gym
import numpy as np
import torch
from ppo_continuous import PPO_continuous
from utils.ContinuesBase import ContinuesBase
import argparse
from torch.utils.tensorboard import SummaryWriter
from normalization import Normalization, RewardScaling

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(torch.version.cuda)  # 打印CUDA版本
print(torch.cuda.is_available())  # 检查CUDA是否可用

local_path = os.path.dirname(__file__)


class Config:
    def __call__(self, *args, **kwargs):
        # Env Params
        parser = argparse.ArgumentParser("Hyperparameters Setting for PPO-continuous")
        parser.add_argument("--env_name", type=str, default="Walker2d-v2", help="env name")
        parser.add_argument("--algo_name", type=str, default="PPO-continuous", help="algorithm name")
        parser.add_argument("--seed", type=int, default=10, help="random seed")
        parser.add_argument("--device", type=str, default='cpu', help="pytorch device")
        # Training Params
        parser.add_argument("--max_train_steps", type=int, default=int(3e6), help=" Maximum number of training steps")
        parser.add_argument("--max_episode_steps", type=int, default=int(1e3), help=" Maximum number of training steps")
        parser.add_argument("--test_eps", type=int, default=20, help=" Maximum number of training steps")
        parser.add_argument("--evaluate_freq", type=float, default=5e3,
                            help="Evaluate the policy every 'evaluate_freq' steps")
        parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
        parser.add_argument("--update_freq", type=int, default=2048, help="Update frequency")
        parser.add_argument("--buffer_size", type=int, default=2048, help="Reply buffer size")
        parser.add_argument("--mini_batch_size", type=int, default=64, help="Minibatch size")
        # Net Params
        parser.add_argument("--hidden_dim", type=int, default=64,
                            help="The number of neurons in hidden layers of the neural network")
        parser.add_argument("--actor_lr", type=float, default=3e-4, help="Learning rate of actor")
        parser.add_argument("--critic_lr", type=float, default=3e-4, help="Learning rate of critic")
        parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
        parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE parameter")
        parser.add_argument("--eps_clip", type=float, default=0.2, help="PPO clip parameter-epsilon-clip")
        parser.add_argument("--k_epochs", type=int, default=10, help="PPO parameter, 更新策略网络的次数")
        parser.add_argument("--entropy_coef", type=float, default=0.01, help="policy entropy")

        # Optim tricks， reference by https://zhuanlan.zhihu.com/p/512327050
        parser.add_argument("--use_state_norm", type=bool, default=False, help="Trick 2:state normalization")
        parser.add_argument("--use_reward_scaling", type=bool, default=True, help="Trick 4:reward scaling")
        parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
        parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
        parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
        parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")

        return parser.parse_args()


def evaluate_policy(args, env, agent: ContinuesBase, state_norm):
    times = 3
    evaluate_reward = 0
    for _ in range(times):
        s, _ = env.reset(seed=args.seed)
        if args.use_state_norm:
            s = state_norm(s, update=False)  # During the evaluating,update=False
        episode_reward = 0
        while True:
            action = agent.sample_action(s, deterministic=True)  # We use the deterministic policy during the evaluating
            s_, r, terminated, truncated, _ = env.step(action)
            if args.use_state_norm:
                s_ = state_norm(s_, update=False)
            episode_reward += r
            s = s_
            if terminated or truncated:
                break
        evaluate_reward += episode_reward

    return evaluate_reward / times


def set_seed(env, env_evaluate, seed=10):
    """ 配置seed """
    if seed == 0:
        return
    env.action_space.seed(seed)
    env_evaluate.action_space.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # config for CPU
    torch.cuda.manual_seed(seed)  # config for GPU
    os.environ['PYTHONHASHSEED'] = str(seed)  # config for python scripts
    return env, env_evaluate


def env_agent_config(cfg):
    """ 配置智能体和环境 """
    env = gym.make(cfg.env_name)  # 创建环境
    env_evaluate = gym.make(cfg.env_name)  # env for evaluate
    env, env_evaluate = set_seed(env, env_evaluate, seed=cfg.seed)
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    max_action = env.action_space.high[0]
    print(f"state dim:{n_states}, action dim:{n_actions}, max action:{max_action}")

    # 更新n_states, max_action和n_actions到cfg参数中
    setattr(cfg, 'n_states', n_states)
    setattr(cfg, 'n_actions', n_actions)
    setattr(cfg, 'max_action', max_action)

    agent = PPO_continuous(cfg)
    return env, env_evaluate, agent


def train(cfg, env: gym.Env, env_evaluate: gym.Env, agent: ContinuesBase):
    """ 训练 """
    print("开始训练！")

    total_steps = 0
    evaluate_num = 0
    sample_count = 0
    evaluate_rewards = []  # 记录每回合的奖励

    # Tensorboard config
    log_dir = os.path.join(local_path,
                           './runs/PPO_continuous/env_{}_number_{}_seed_{}'.format(cfg.env_name, 1, cfg.seed))
    writer = SummaryWriter(log_dir=log_dir)

    state_norm = Normalization(shape=cfg.n_states)  # Trick 2:state normalization

    while total_steps < cfg.max_train_steps:
        s, _ = env.reset(seed=cfg.seed)  # 重置环境，返回初始状态
        ep_step = 0
        while True:
            ep_step += 1
            sample_count += 1
            a, a_logprob = agent.sample_action(s)  # 选择动作
            s_, r, terminated, truncated, _ = env.step(a)  # 更新环境，返回transition

            if ep_step == cfg.max_episode_steps:
                truncated = True

            # 保存transition
            agent.memory.push((s, a, a_logprob, s_, r, terminated, truncated))
            s = s_  # 更新下一个状态
            total_steps += 1

            # update policy every n steps
            if sample_count % cfg.buffer_size == 0:
                agent.update(total_steps)

            if total_steps % cfg.evaluate_freq == 0:
                evaluate_num += 1
                evaluate_reward = evaluate_policy(cfg, env_evaluate, agent, state_norm)
                evaluate_rewards.append(evaluate_reward)
                print("evaluate_num:{} \t evaluate_reward:{} \t".format(evaluate_num, evaluate_reward))
                writer.add_scalar('step_rewards_{}'.format(cfg.env_name), evaluate_rewards[-1], global_step=total_steps)
                # Save the rewards
                if evaluate_num % cfg.save_freq == 0:
                    model_dir = os.path.join(local_path,
                                             './data_train/PPO_continuous_env_{}_number_{}_seed_{}.npy'.format(
                                                 cfg.env_name, 1, 10))
                    np.save(model_dir, np.array(evaluate_rewards))
            if terminated or truncated:
                break

    print("完成训练！")
    env.close()
    return agent, {'rewards': evaluate_rewards}


if __name__ == '__main__':
    cfg = Config().__call__()
    env, env_evaluate, agent = env_agent_config(cfg)
    best_agent, res_dic = train(cfg, env, env_evaluate, agent)

import copy
import gym
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import random
import numpy as np
from agent import PPO_continuous
import argparse
from torch.utils.tensorboard import SummaryWriter

print(torch.version.cuda)  # 打印CUDA版本
print(torch.cuda.is_available())  # 检查CUDA是否可用

class Config:
    @property
    def args(self):
        parser = argparse.ArgumentParser("Hyperparameters Setting for PPO-continuous")
        parser.add_argument("--env_name", type=str, default="Walker2d-v2", help="env name")
        parser.add_argument("--algo_name", type=str, default="PPO-continuous", help="algorithm name")
        parser.add_argument("--seed", type=int, default=1, help="random seed")
        parser.add_argument("--device", type=str, default='cuda', help="pytorch device")
        parser.add_argument("--max_steps", type=int, default=int(3e6), help=" Maximum number of training steps")
        parser.add_argument("--max_episode_steps", type=int, default=int(1e3), help=" Maximum number of training steps")
        parser.add_argument("--train_eps", type=int, default=int(3e3), help=" Maximum number of training steps")
        parser.add_argument("--test_eps", type=int, default=20, help=" Maximum number of training steps")
        parser.add_argument("--evaluate_freq", type=float, default=5e3, help="Evaluate the policy every 'evaluate_freq' steps")
        parser.add_argument("--eval_eps", type=float, default=5, help="评估的回合数")
        parser.add_argument("--eval_per_episode", type=float, default=10, help="评估的频率")
        parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
        parser.add_argument("--update_freq", type=int, default=100, help="Update frequency")
        parser.add_argument("--policy_dist", type=str, default="Gaussian", help="Beta or Gaussian")
        parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
        parser.add_argument("--mini_batch_size", type=int, default=64, help="Minibatch size")
        parser.add_argument("--actor_hidden_dim", type=int, default=64, help="The number of neurons in hidden layers of the neural network")
        parser.add_argument("--critic_hidden_dim", type=int, default=64, help="The number of neurons in hidden layers of the neural network")
        parser.add_argument("--actor_lr", type=float, default=3e-4, help="Learning rate of actor")
        parser.add_argument("--critic_lr", type=float, default=3e-4, help="Learning rate of critic")
        parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor, 折扣因子")
        parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
        parser.add_argument("--eps_clip", type=float, default=0.2, help="PPO clip parameter-epsilon-clip")
        parser.add_argument("--k_epochs", type=int, default=10, help="PPO parameter, 更新策略网络的次数")

        # Optim 10 tricks， reference by https://zhuanlan.zhihu.com/p/512327050
        parser.add_argument("--use_state_norm", type=bool, default=True, help="Trick 2:state normalization")
        parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
        parser.add_argument("--use_reward_scaling", type=bool, default=True, help="Trick 4:reward scaling")
        parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
        parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
        parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
        parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
        parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
        parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")

        args = parser.parse_args()
        return args


def all_seed(env, seed = 1):
    """ 配置seed """
    if seed == 0:
        return
    env.seed(seed) # env config
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed) # config for CPU
    torch.cuda.manual_seed(seed) # config for GPU
    os.environ['PYTHONHASHSEED'] = str(seed) # config for python scripts
    # config for cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

def env_agent_config(cfg):
    """ 配置智能体和环境 """
    env = gym.make(cfg.env_name) # 创建环境
    all_seed(env, seed=cfg.seed)
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]
    max_ep_steps = env._max_episode_steps
    print(f"max_ep_steps:{max_ep_steps}")
    print(f"状态空间维度：{n_states}，动作空间维度：{n_actions}")
    # 更新n_states和n_actions到cfg参数中
    setattr(cfg, 'n_states', n_states)
    setattr(cfg, 'n_actions', n_actions) 
    agent = PPO_continuous(cfg)
    return env, agent

def train(cfg, env, agent):
    ''' 训练 '''
    print("开始训练！")
    rewards = []  # 记录每回合的奖励
    steps = [] # 记录每回合的步数
    best_ep_reward = 0 # 记录最大回合奖励
    output_agent = None

    writer = SummaryWriter(log_dir='runs/PPO_continuous/env_{}_{}_number_{}_seed_{}'.format(cfg.env_name, cfg.policy_dist, 1, cfg.seed))

    for i_ep in range(cfg.train_eps):
        ep_reward = 0  # 记录一回合内的奖励
        ep_step = 0
        state = env.reset()  # 重置环境，返回初始状态
        while ep_step < cfg.max_episode_steps:
            ep_step += 1
            action = agent.sample_action(state)  # 选择动作
            next_state, reward, done, _ = env.step(action)  # 更新环境，返回transition
            agent.memory.push((state, action, agent.log_probs, reward, done))  # 保存transition
            state = next_state  # 更新下一个状态
            agent.update()  # 更新智能体
            ep_reward += reward  # 累加奖励
            if done:
                break

        if (i_ep+1) % cfg.eval_per_episode == 0:
            sum_eval_reward = 0
            for _ in range(cfg.eval_eps):
                eval_ep_reward = 0
                state = env.reset()
                for _ in range(cfg.max_episode_steps):
                    action = agent.predict_action(state)  # 选择动作
                    next_state, reward, done, _ = env.step(action)  # 更新环境，返回transition
                    state = next_state  # 更新下一个状态
                    eval_ep_reward += reward  # 累加奖励
                    if done:
                        break
                sum_eval_reward += eval_ep_reward
            mean_eval_reward = sum_eval_reward/cfg.eval_eps
            if mean_eval_reward >= best_ep_reward:
                best_ep_reward = mean_eval_reward
                output_agent = copy.deepcopy(agent)
                print(f"回合：{i_ep+1}/{cfg.train_eps}，奖励：{ep_reward:.2f}，评估奖励：{mean_eval_reward:.2f}，最佳评估奖励：{best_ep_reward:.2f}，更新agent")
            else:
                print(f"回合：{i_ep+1}/{cfg.train_eps}，奖励：{ep_reward:.2f}，评估奖励：{mean_eval_reward:.2f}，最佳评估奖励：{best_ep_reward:.2f}")
        steps.append(ep_step)
        rewards.append(ep_reward)
    print("完成训练！")
    env.close()
    return output_agent,{'rewards':rewards}

def test(cfg, env, agent):
    print("开始测试！")
    rewards = []  # 记录所有回合的奖励
    steps = []
    for i_ep in range(cfg.test_eps):
        ep_reward = 0  # 记录一回合内的奖励
        ep_step = 0
        state = env.reset()  # 重置环境，返回初始状态
        for _ in range(cfg.max_episode_steps):
            ep_step+=1
            action = agent.predict_action(state)  # 选择动作
            next_state, reward, done, _ = env.step(action)  # 更新环境，返回transition
            state = next_state  # 更新下一个状态
            ep_reward += reward  # 累加奖励
            if done:
                break
        steps.append(ep_step)
        rewards.append(ep_reward)
        print(f"回合：{i_ep+1}/{cfg.test_eps}，奖励：{ep_reward:.2f}")
    print("完成测试")
    env.close()
    return {'rewards':rewards}

def smooth(data, weight=0.9):  
    '''用于平滑曲线，类似于Tensorboard中的smooth曲线
    '''
    last = data[0] 
    smoothed = []
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point  # 计算平滑值
        smoothed.append(smoothed_val)                    
        last = smoothed_val                                
    return smoothed

def plot_rewards(rewards,cfg, tag='train'):
    ''' 画图
    '''
    sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.title(f"{tag}ing curve on {cfg.device} of {cfg.algo_name} for {cfg.env_name}")
    plt.xlabel('epsiodes')
    plt.plot(rewards, label='rewards')
    plt.plot(smooth(rewards), label='smoothed')
    plt.legend()
    plt.show()

if __name__=='__main__':
    # 获取参数
    cfg = Config().args
    # 训练
    env, agent = env_agent_config(cfg)
    best_agent, res_dic = train(cfg, env, agent)
    plot_rewards(res_dic['rewards'], cfg, tag="train")  
    # 测试
    res_dic = test(cfg, env, best_agent)
    plot_rewards(res_dic['rewards'], cfg, tag="test")  # 画出结果

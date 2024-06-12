import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import logging
import json
import h5py
import robopal
from robopal.wrappers import GoalEnvWrapper
from PPO.ppo_continuous import PPO_continuous
import torch
from utils.normalization import Normalization
import time

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
local_path = os.path.dirname(__file__)
log_path = os.path.join(local_path, 'log')


def roll_out(agent, env, n_episodes):
    return_list = []
    for _ in range(n_episodes):
        obs, info = env.reset()
        for _ in range(500):
            action = agent.take_action(obs["observation"])
            next_obs, reward, termination, truncation, info = env.step(action)
            obs = next_obs
            if info["is_success"]:
                break
        return_list.append(int(info["is_success"]))
    return np.mean(return_list)


def data_loader(data_path):

    # read .hdf5 files
    file = h5py.File(data_path,'r')   
    logging.info("Reading from {}".format(data_path))

    env_args = json.loads(file["data"].attrs["env_args"])

    demos = list(file["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    expert_obs = []
    expert_a = []
    # playback each episode
    for ep in demos:
        logging.info("\n>Reading episode: {}".format(ep))
        expert_obs.append(file["data/{}/obs/low_dim".format(ep)][()])
        expert_a.append(file["data/{}/actions".format(ep)][()])

    logging.info("Data loaded successfully!")
    return expert_obs, expert_a, env_args


def make_env(env_args):
    logging.info(f"env name: {env_args["env_name"]}")
    logging.info(f"env meta: {env_args["env_kwargs"]}")

    env = robopal.make(
        env_args["env_name"],
        **env_args["env_kwargs"]
    )
    env = GoalEnvWrapper(env)
    return env


def test(args):
    _, _, env_args = data_loader(args.data_path)
    env = make_env(env_args)
    
    setattr(args, "n_states", env.observation_space["observation"].shape[0])
    setattr(args, "n_actions", env.action_space.shape[0])
    setattr(args, 'max_action', env.action_space.high[0])

    torch.manual_seed(0)
    np.random.seed(0)

    generator = PPO_continuous(args)

    epoch = 4900
    model_dir = os.path.join(log_path, f'./data_train')
    model_path = os.path.join(model_dir, f'GAILPPO_{epoch}.pth')
    actor_state_dict = torch.load(model_path)
    generator.actor.load_state_dict(actor_state_dict)

    if args.use_state_norm:
        state_norm = Normalization(shape=args.n_states)
        state_norm.load(os.path.join(model_dir, f'state_norm_{epoch}.npy'))

    n_episodes = 20
    generator.actor.eval()
    generator.critic.eval()
    for ep in range(n_episodes):
        episode_return = 0
        obs, _ = env.reset()
        if args.use_state_norm:
            obs["observation"] = state_norm(obs["observation"], update=False)
        while True:
            action = generator.sample_action(obs["observation"], deterministic=True)

            next_obs, reward, termination, truncation, _ = env.step(action)

            if args.use_state_norm:
                next_obs["observation"] = state_norm(next_obs["observation"], update=False)

            obs = next_obs
            episode_return += reward

            time.sleep(0.01)
            if termination or truncation:
                break


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--data_path", type=str, default="/home/ubuntu/zhr/robopal/robopal/collections/collections_1717654238_740447/demo.hdf5")

    # PPO
    # Env Params
    args.add_argument("--env_name", type=str, default="Walker2d-v4", help="env name")
    args.add_argument("--algo_name", type=str, default="PPO-continuous", help="algorithm name")
    args.add_argument("--seed", type=int, default=10, help="random seed")
    args.add_argument("--device", type=str, default=device, help="pytorch device")
    # Training Params
    args.add_argument("--max_train_steps", type=int, default=int(3e6), help=" Maximum number of training steps")
    args.add_argument("--max_episode_horizon", type=int, default=int(1e3), help=" Maximum number of training steps")
    args.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    args.add_argument("--update_freq", type=int, default=512, help="Update frequency")
    args.add_argument("--buffer_size", type=int, default=4096, help="Reply buffer size")
    args.add_argument("--mini_batch_size", type=int, default=1024, help="Minibatch size")
    # Net Params
    args.add_argument("--hidden_dim", type=int, default=256,
                        help="The number of neurons in hidden layers of the neural network")
    args.add_argument("--lr", type=float, default=1e-4, help="Learning rate in PPO")
    args.add_argument("--lr_discrim", type=float, default=1e-4, help="Learning rate in Discriminator")
    args.add_argument("--gamma", type=float, default=0.97, help="Discount factor")
    args.add_argument("--gae_lambda", type=float, default=0.98, help="GAE parameter")
    args.add_argument("--eps_clip", type=float, default=0.2, help="PPO clip parameter-epsilon-clip")
    args.add_argument("--k_epochs", type=int, default=10, help="PPO parameter, 更新策略网络的次数")
    args.add_argument("--entropy_coef", type=float, default=1e-2, help="policy entropy")
    args.add_argument("--max_grad_norm", type=float, default=0.5, help="max_grad_norm")

    # Optim tricks， reference by https://zhuanlan.zhihu.com/p/512327050
    args.add_argument("--use_state_norm", type=bool, default=True, help="Trick 2:state normalization")
    args.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    args.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")

    # DISCRIMINATOR
    args.add_argument('--suspend_accu_exp', type=float, default=0.8,
                    help='accuracy for suspending discriminator about expert data (default: 0.8)')
    args.add_argument('--suspend_accu_gen', type=float, default=0.9,
                    help='accuracy for suspending discriminator about generated data (default: 0.8)')

    args = args.parse_args()

    test(args)

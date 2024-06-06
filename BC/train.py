import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import logging
import json
import h5py
import numpy as np
import torch
import robopal
from robopal.wrappers import GoalEnvWrapper
from BC.bc import BehaviorClone


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


def train(args):
    expert_obs, expert_a, env_args = data_loader(args.data_path)
    env = make_env(env_args)

    # env.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)

    expert_obs = np.concatenate(expert_obs, axis=0)
    expert_a = np.concatenate(expert_a, axis=0)
    obs_dim = expert_obs.shape[1]
    action_dim = expert_a.shape[1]
    bc_agent = BehaviorClone(
        obs_dim,
        action_dim, 
        args.hidden_dim, 
        args.lr
    )
    n_iterations = 20000
    batch_size = 256

    loss_list = []
    with tqdm(total=n_iterations, desc="进度条") as pbar:
        for i in range(n_iterations):
            sample_indices = np.random.randint(low=0,
                                            high=expert_obs.shape[0],
                                            size=batch_size)

            loss = bc_agent.learn(expert_obs[sample_indices], expert_a[sample_indices])
            
            if (i + 1) % 100 == 0:
                print(f"Iteration {i + 1}, loss: {loss}")
                loss_list.append(loss)
                success = roll_out(bc_agent, env, 1)
                pbar.set_postfix({'return': '%.3f' % np.mean(success)})
            pbar.update(1)

    iteration_list = list(range(len(loss_list)))
    plt.plot(iteration_list, loss_list)
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.show()


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--data_path", type=str, default="/home/ubuntu/zhr/robopal/robopal/collections/collections_1717654238_740447/demo.hdf5")
    args.add_argument("--hidden_dim", type=str, default=128)
    args.add_argument("--lr", type=float, default=1e-3)
    args = args.parse_args()

    train(args)

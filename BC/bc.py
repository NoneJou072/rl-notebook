import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import logging
import json
import h5py
import robopal
from robopal.wrappers import GoalEnvWrapper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        """
        初始化Q网络为全连接网络
        """
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)  # 输入层
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # 隐藏层
        self.fc3 = nn.Linear(hidden_dim, action_dim)  # 输出层

    def forward(self, x):
        # 各层对应的激活函数
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    

class BehaviorClone:
    def __init__(self, state_dim, action_dim, hidden_dim, lr):
        self.policy = MLP(state_dim, action_dim, hidden_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def learn(self, states, actions):
        states = torch.tensor(states, dtype=torch.float).to(device)
        actions = torch.tensor(actions).to(device)
        actions_ = self.policy(states)

        bc_loss = F.mse_loss(actions_, actions)

        self.optimizer.zero_grad()
        bc_loss.backward()
        self.optimizer.step()

        return bc_loss.item()

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(device)
        action = self.policy(state)
        return action.flatten().cpu().detach().numpy()


def roll_out(agent, env, n_episodes):
    return_list = []
    for _ in range(n_episodes):
        obs, info = env.reset()
        for _ in range(1000):
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
            
            if (i + 1) % 2 == 0:
                # print(f"Iteration {i + 1}, loss: {loss}")
                loss_list.append(loss)
                # success = roll_out(bc_agent, env, 1)
                # pbar.set_postfix({'return': '%.3f' % np.mean(success)})
            pbar.update(1)

    iteration_list = list(range(len(loss_list)))
    plt.plot(iteration_list, loss_list)
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.show()


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--data_path", type=str, default="./data")
    args.add_argument("--hidden_dim", type=str, default=128)
    args.add_argument("--lr", type=float, default=1e-3)
    args = args.parse_args()

    train(args)

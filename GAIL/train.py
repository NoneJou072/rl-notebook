import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import shutil
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from tqdm import tqdm
import argparse
import logging
import json
import h5py
import robopal
from robopal.wrappers import GoalEnvWrapper
from GAIL.gail import GAIL
from PPO.ppo_continuous import PPO_continuous
import torch
from utils.normalization import Normalization

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
local_path = os.path.dirname(__file__)
log_path = os.path.join(local_path, 'log')


def roll_out(args, agent, env, n_episodes, state_norm=None):
    return_list = []
    for _ in range(n_episodes):
        obs, info = env.reset()
        if args.use_state_norm:
            obs["observation"] = state_norm(obs["observation"], update=False)
        for _ in range(100):
            action = agent.sample_action(obs["observation"], deterministic=True)
            next_obs, reward, termination, truncation, info = env.step(action)
            if args.use_state_norm:
                next_obs["observation"] = state_norm(next_obs["observation"], update=False)
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

    env_args["env_kwargs"]["render_mode"] = None
    env = robopal.make(
        env_args["env_name"],
        **env_args["env_kwargs"]
    )
    env = GoalEnvWrapper(env)
    return env


def train(args):
    expert_obs, expert_a, env_args = data_loader(args.data_path)
    env = make_env(env_args)
    
    setattr(args, "n_states", env.observation_space["observation"].shape[0])
    setattr(args, "n_actions", env.action_space.shape[0])
    setattr(args, 'max_action', env.action_space.high[0])

    torch.manual_seed(0)
    np.random.seed(0)

    # Tensorboard config
    log_dir = os.path.join(log_path, f'./runs/GAIL')
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    expert_obs = np.concatenate(expert_obs, axis=0)
    expert_a = np.concatenate(expert_a, axis=0)

    if args.use_state_norm:
        state_norm = Normalization(shape=args.n_states)
        expert_state_norm = Normalization(shape=args.n_states)
        for i in range(expert_obs.shape[0]):
            expert_obs[i] = expert_state_norm(expert_obs[i])

    generator = PPO_continuous(args)
    discriminator = GAIL(args)

    train_discrim_flag = True
    discrim_loss, expert_acc, learner_acc, discrim_ep_score = 0, 0, 0, 0

    total_timesteps = 0
    for epoch in range(args.n_epochs):
        
        discrim_ep_score = 0
        return_list = []
        generator.memory.clear()
        generator.actor.eval()
        generator.critic.eval()
        discriminator.discriminator.eval()

        rollout_success = roll_out(args, generator, env, 1, state_norm)

        with tqdm(total=args.n_episodes, desc=f">Epoch: {epoch}") as pbar:
            for ep in range(args.n_episodes):
                episode_return = 0
                obs, _ = env.reset()
                if args.use_state_norm:
                    obs["observation"] = state_norm(obs["observation"])

                for ep_step in range(args.max_episode_horizon):
                    action, a_logprob = generator.sample_action(obs["observation"])
                    next_obs, reward, _, _, info = env.step(action)
                    
                    if args.use_state_norm:
                        next_obs["observation"] = state_norm(next_obs["observation"])

                    irl_reward = discriminator.get_reward(obs["observation"], action)

                    termination = info["is_success"]
                    truncation = False
                    if ep_step == args.max_episode_horizon - 1:
                        truncation = True

                    generator.memory.push((
                        obs["observation"], action, a_logprob, next_obs["observation"], irl_reward, termination, truncation
                    ))

                    obs = next_obs
                    episode_return += reward
                    discrim_ep_score += irl_reward
                    total_timesteps += 1

                return_list.append(episode_return)
                pbar.set_postfix({'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(total_timesteps)

        # START TRAINING
        generator.actor.train()
        generator.critic.train()
        discriminator.discriminator.train()

        if train_discrim_flag:
            discrim_loss, expert_acc, learner_acc = discriminator.learn(expert_obs, expert_a, generator.memory)
            print("Discriminator loss: %.2f, Expert: %.2f%% | Learner: %.2f%% | ep_score: %.2f" % (
                discrim_loss, expert_acc * 100, learner_acc * 100, discrim_ep_score / args.n_episodes)
            )
            # if expert_acc > args.suspend_accu_exp and learner_acc > args.suspend_accu_gen:
            #     train_discrim_flag = False

        actor_loss, critic_loss, lr = generator.update(total_timesteps)

        if epoch % args.save_freq == 0: 
            # Save the Actor weights
            model_dir = os.path.join(log_path, f'./data_train')
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            torch.save(generator.actor.state_dict(), os.path.join(model_dir, f'GAILPPO_{epoch}.pth'))
            if args.use_state_norm:
                state_norm.save(os.path.join(model_dir, f'state_norm_{epoch}.npy'))
            print("Model saved!")
        
        writer.add_scalar('discriminator/discrim_loss', discrim_loss, global_step=epoch)
        writer.add_scalar('discriminator/expert_acc', expert_acc, global_step=epoch)
        writer.add_scalar('discriminator/learner_acc', learner_acc, global_step=epoch)
        writer.add_scalar('generator/critic_loss', critic_loss, global_step=epoch)
        writer.add_scalar('generator/actor_loss', actor_loss, global_step=epoch)
        writer.add_scalar('generator/rollout_success', rollout_success, global_step=epoch)
        writer.add_scalar('generator/discrim_ep_score', discrim_ep_score / args.n_episodes, global_step=epoch)
        writer.add_scalar('generator/lr', lr, global_step=epoch)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--data_path", type=str, default="/home/ubuntu/zhr/robopal/robopal/collections/collections_1718681289_7812529/demo.hdf5")

    # PPO
    # Env Params
    args.add_argument("--env_name", type=str, default="Walker2d-v4", help="env name")
    args.add_argument("--algo_name", type=str, default="PPO-continuous", help="algorithm name")
    args.add_argument("--seed", type=int, default=10, help="random seed")
    args.add_argument("--device", type=str, default=device, help="pytorch device")
    # Training Params
    args.add_argument("--n_epochs", type=int, default=int(2e3), help="Maximum number of training steps")
    args.add_argument("--n_episodes", type=int, default=10, help="Maximum number of training steps")
    args.add_argument("--max_train_steps", type=int, default=int(2e7), help="Maximum number of training steps")
    args.add_argument("--max_episode_horizon", type=int, default=int(200), help="Maximum number of training steps")
    args.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    args.add_argument("--buffer_size", type=int, default=2560, help="Reply buffer size")
    args.add_argument("--mini_batch_size", type=int, default=64, help="Minibatch size")
    # Net Params
    args.add_argument("--hidden_dim", type=int, default=256, help="The number of neurons in hidden layers of the neural network")
    args.add_argument("--lr", type=float, default=3e-4, help="Learning rate in PPO")
    args.add_argument("--lr_discrim", type=float, default=3e-4, help="Learning rate in Discriminator")
    args.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    args.add_argument("--gae_lambda", type=float, default=0.95, help="GAE parameter")
    args.add_argument("--eps_clip", type=float, default=0.2, help="PPO clip parameter-epsilon-clip")
    args.add_argument("--k_epochs", type=int, default=10, help="PPO parameter, 更新策略网络的次数")
    args.add_argument("--entropy_coef", type=float, default=1e-2, help="policy entropy")
    args.add_argument("--max_grad_norm", type=float, default=1.0, help="max_grad_norm")

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

    train(args)

import os
import argparse
from copy import deepcopy
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from iorl import IORL
from utils.ModelBase import ModelBase
from utils.replay_buffer import Trajectory
from envs import DrawerBox, GoalEnvWrapper

local_path = os.path.dirname(__file__)
log_path = os.path.join(local_path, './log')


def args():
    parser = argparse.ArgumentParser("Hyperparameters Setting for IORL")
    parser.add_argument("--env_name", type=str, default="DrawerBox-v1", help="env name")
    parser.add_argument("--algo_name", type=str, default="IORL", help="algorithm name")
    parser.add_argument("--seed", type=int, default=10, help="random seed")
    parser.add_argument("--device", type=str, default='cuda:0', help="pytorch device")
    # Training Params
    parser.add_argument("--max_train_steps", type=int, default=int(1e7), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=2e3,
                        help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=5, help="Save frequency")
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
    parser.add_argument("--update_freq", type=int, default=40, help="Take 50 steps,then update the networks 50 times")
    parser.add_argument("--k_future", type=int, default=4, help="Her k future")
    parser.add_argument("--sigma", type=int, default=0.2, help="The std of Gaussian noise for exploration")
    parser.add_argument("--k_update", type=bool, default=2, help="Delayed policy update frequence")

    return parser.parse_args()


class IORLModel(ModelBase):
    def __init__(self, env, args):
        super().__init__(env, args)
        self.agent = IORL(env, args)
        self.model_name = f'{self.agent.agent_name}_{self.args.env_name}_num_{1}_seed_{self.args.seed}'
        self.random_steps = args.random_steps
        self.update_freq = args.update_freq
        self.max_train_steps = args.max_train_steps

    def train(self):
        """ 训练 """
        print("开始训练！")
        total_steps = 0
        # Tensorboard config
        log_dir = os.path.join(log_path, f'./runs/{self.model_name}')
        if os.path.exists(log_dir):
            i = 1
            while os.path.exists(log_dir + '_' + str(i)):
                i += 1
            log_dir = log_dir + '_' + str(i)

        writer = SummaryWriter(log_dir=log_dir)

        while total_steps < self.max_train_steps:
            # Task1
            self.env.env.TASK_FLAG = 0
            obs, _ = self.env.reset()
            traj_draw = Trajectory()
            while True:
                a = self.agent.sample_action(obs, task='drawer', deterministic=False)  # 选择动作
                obs_, r, terminated, truncated, info = self.env.step(a)  # 更新环境，返回transition
                traj_draw.push((obs["observation"], a, obs_["observation"], r, obs["achieved_goal"], obs["desired_goal"],
                               obs_["achieved_goal"]))
                obs = deepcopy(obs_)

                total_steps += 1
                if truncated:
                    break

            # Task2
            self.env.env.TASK_FLAG = 1
            obs, _ = self.env.reset()
            traj_place = Trajectory()
            while True:
                a = self.agent.sample_action(obs, task='place', deterministic=False)  # 选择动作
                obs_, r, terminated, truncated, info = self.env.step(a)  # 更新环境，返回transition
                traj_place.push((obs["observation"], a, obs_["observation"], r, obs["achieved_goal"],
                                obs["desired_goal"], obs_["achieved_goal"]))
                obs = deepcopy(obs_)

                total_steps += 1
                if truncated:
                    break

            # 将轨迹分割成子任务1和子任务2的，其中，子任务2使用完整的轨迹
            traj_reach = Trajectory()
            for transition in traj_draw.buffer:
                traj_reach.push(transition)
                reached = self.agent.check_reached(transition[6][:3], transition[5][:3])
                if reached:
                    break
            self.agent.memory_draw.push(traj_draw)
            self.agent.memory_reach.push(traj_reach)

            traj_reach = Trajectory()
            for transition in traj_place.buffer:
                traj_reach.push(transition)
                reached = self.agent.check_reached(transition[6][:3], transition[5][:3])
                if reached:
                    break
            self.agent.memory_place.push(traj_place)
            self.agent.memory_reach.push(traj_reach)

            if total_steps >= self.random_steps:
                for _ in range(self.update_freq):
                    self.agent.update()
                self.agent.update_target_net()

            if total_steps >= self.random_steps and total_steps % self.args.evaluate_freq == 0:
                evaluate_reward_drawer, evaluate_reward_place, success_rate_drawer, success_rate_place = self.evaluate_policy()
                print(
                    f"total_steps:{total_steps} \t evaluate_reward_draw:{evaluate_reward_drawer} \t \
                    evaluate_reward_place:{evaluate_reward_place} \t success_rate_drawer:{success_rate_drawer}\t \
                    success_rate_place:{success_rate_place}"
                )
                writer.add_scalar('step_rewards_{}_drawer'.format(self.args.env_name), evaluate_reward_drawer,
                                  global_step=total_steps)
                writer.add_scalar('success_rate_{}_drawer'.format(self.args.env_name), success_rate_drawer,
                                  global_step=total_steps)
                writer.add_scalar('critic_loss_{}'.format(self.args.env_name), self.agent.critic_loss_record,
                                  global_step=total_steps)
                writer.add_scalar('actor_loss_{}'.format(self.args.env_name), self.agent.actor_loss_record,
                                  global_step=total_steps)
                # Save Actor weights
                model_dir = os.path.join(log_path, f'./data_train/{self.agent.agent_name}')
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)
                torch.save(self.agent.actor.state_dict(), os.path.join(model_dir, f'{self.model_name}.pth'))

        print("完成训练！")
        self.env.close()

    def evaluate_policy(self):
        times = 10
        # evaluate drawer stage
        self.env.env.TASK_FLAG = 0
        evaluate_reward_drawer = 0
        success_rate_drawer = 0
        for _ in range(times):
            episode_reward = 0
            obs, _ = self.env_evaluate.reset()
            while True:
                # We use the deterministic policy during the evaluating
                action = self.agent.sample_action(obs, task='drawer', deterministic=True)
                obs_, r, terminated, truncated, info = self.env_evaluate.step(action)
                obs = deepcopy(obs_)
                episode_reward += r
                if truncated:
                    success_rate_drawer += info['is_drawer_success']
                    break
            evaluate_reward_drawer += episode_reward

        # # evaluate drawer stage
        # evaluate_reward_reach = 0
        # for _ in range(times):
        #     episode_reward = 0
        #     obs, _ = self.env_evaluate.reset()
        #     while np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"]) <= 0.05:
        #         obs, _ = self.env.reset()
        #     while True:
        #         obs['desired_goal'] *= 0
        #         s = torch.unsqueeze(torch.tensor(obs['observation'], dtype=torch.float32), 0).to(self.agent.device)
        #         g = torch.unsqueeze(torch.tensor(obs['desired_goal'], dtype=torch.float32), 0).to(self.agent.device)
        #         ag = torch.unsqueeze(torch.tensor(obs['achieved_goal'], dtype=torch.float32), 0).to(self.agent.device)
        #         action = self.agent.actor(s, g, ag).data.cpu().numpy().flatten()
        #         obs, r, terminated, truncated, _ = self.env_evaluate.step(action)
        #         r = self.env.compute_reward(obs['observation'][:3], obs['achieved_goal'], None)
        #         episode_reward += r
        #         if truncated:
        #             break
        #     evaluate_reward_reach += episode_reward

        # evaluate place stage
        self.env.env.TASK_FLAG = 1
        evaluate_reward_place = 0
        success_rate_place = 0
        for _ in range(times):
            episode_reward = 0
            obs, _ = self.env_evaluate.reset()
            while True:
                action = self.agent.sample_action(obs, task='place', deterministic=True)
                obs, r, terminated, truncated, info = self.env_evaluate.step(action)
                episode_reward += r
                if truncated:
                    success_rate_place += info['is_place_success']
                    break
            evaluate_reward_place += episode_reward

        return evaluate_reward_drawer / times, evaluate_reward_place / times, success_rate_drawer / times, success_rate_place / times


def make_env(args):
    """ 配置环境 """
    env = DrawerBox(is_render=False)
    env = GoalEnvWrapper(env)

    state_dim = env.observation_space.spaces["observation"].shape[0]
    action_dim = env.action_space.shape[0]
    goal_dim = env.observation_space.spaces["desired_goal"].shape[0]
    max_action = float(env.action_space.high[0])
    min_action = float(env.action_space.low[0])

    print(f"state dim:{state_dim}, action dim:{action_dim}, max_epi_steps:{env.max_episode_steps}")
    print(f"max action:{max_action}, min action:{min_action}")

    setattr(args, 'state_dim', state_dim)
    setattr(args, 'action_dim', action_dim)
    setattr(args, 'goal_dim', goal_dim)
    setattr(args, 'max_episode_steps', env.max_episode_steps)
    setattr(args, 'max_action', max_action)

    return env


if __name__ == '__main__':
    args = args()
    env = make_env(args)
    model = IORLModel(
        env=env,
        args=args,
    )
    model.train()

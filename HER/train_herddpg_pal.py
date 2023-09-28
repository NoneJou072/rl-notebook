import os
import torch
from copy import deepcopy
import shutil
from HER.HERDDPG import HERDDPG
from utils.ModelBase import ModelBase
import argparse
from torch.utils.tensorboard import SummaryWriter
from utils.replay_buffer import Trajectory

from robopal.demos.demo_pick_place import PickAndPlaceEnv
from robopal.commons.gym_wrapper import GoalEnvWrapper as GymWrapper
from gymnasium.utils import env_checker


local_path = os.path.dirname(__file__)
log_path = os.path.join(local_path, 'log')


def args():
    parser = argparse.ArgumentParser("Hyperparameters Setting for DDPG")
    parser.add_argument("--env_name", type=str, default="FetchPickAndPlace-v2", help="env name")
    parser.add_argument("--algo_name", type=str, default="HERDDPG", help="algorithm name")
    parser.add_argument("--seed", type=int, default=10, help="random seed")
    parser.add_argument("--device", type=str, default='cuda:0', help="pytorch device")
    # Training Params
    parser.add_argument("--max_train_steps", type=int, default=int(1e7), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=5e3,
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
    parser.add_argument("--update_freq", type=int, default=50, help="Take 50 steps,then update the networks 50 times")
    parser.add_argument("--k_future", type=int, default=4, help="Her k future")

    return parser.parse_args()


class HERDDPGModel(ModelBase):
    def __init__(self, env, args):
        super().__init__(env, args)
        self.agent = HERDDPG(env, args)
        self.model_name = f'{self.agent.agent_name}_{self.args.env_name}_num_{2}_seed_{self.args.seed}'
        self.random_steps = args.random_steps
        self.update_freq = args.update_freq

    def train(self):
        """ 训练 """
        print("开始训练！")
        total_steps = 0
        # Tensorboard config
        log_dir = os.path.join(log_path, f'./runs/{self.model_name}')
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        writer = SummaryWriter(log_dir=log_dir)

        while total_steps < self.args.max_train_steps:
            traj = Trajectory()
            obs, _ = self.env.reset()  # 重置环境，返回初始状态
            # begin a new episode
            while True:
                a = self.agent.sample_action(obs["observation"], obs["desired_goal"])  # 选择动作
                obs_, r, terminated, truncated, _ = self.env.step(a)  # 更新环境，返回transition

                traj.push((obs["observation"], a, obs_["observation"], r, obs["achieved_goal"], obs["desired_goal"], obs_["achieved_goal"]))

                obs = deepcopy(obs_)

                # Take 50 steps,then update the networks 50 times
                total_steps += 1
                if total_steps >= self.random_steps and total_steps % self.update_freq == 0:
                    for _ in range(self.update_freq):
                        self.agent.update()
                    self.agent.update_target_net()

                if total_steps >= self.random_steps and total_steps % self.args.evaluate_freq == 0:
                    evaluate_reward = self.evaluate_policy()
                    print(f"timestep:{total_steps} \t evaluate_reward:{evaluate_reward} \t")

                    writer.add_scalar('step_rewards_{}'.format(self.args.env_name), evaluate_reward,
                                      global_step=total_steps)
                    writer.add_scalar('critic_loss_{}'.format(self.args.env_name), self.agent.critic_loss_record,
                                      global_step=total_steps)
                    writer.add_scalar('actor_loss_{}'.format(self.args.env_name), self.agent.actor_loss_record,
                                      global_step=total_steps)

                    # Save the Actor weights
                    model_dir = os.path.join(log_path, f'./data_train/{self.agent.agent_name}')
                    if not os.path.exists(model_dir):
                        os.makedirs(model_dir)
                    torch.save(self.agent.actor.state_dict(), os.path.join(model_dir, f'{self.model_name}.pth'))
                if truncated:
                    break

            self.agent.memory.push(traj)

        print("完成训练！")
        self.env.close()

    def evaluate_policy(self):
        times = 5
        evaluate_reward = 0
        for _ in range(times):
            obs, _ = self.env_evaluate.reset()
            if self.args.use_state_norm:
                s = self.state_norm(obs["observation"], update=False)  # During the evaluating,update=False
            episode_reward = 0
            while True:
                # We use the deterministic policy during the evaluating
                action = self.agent.sample_action(obs["observation"], obs["desired_goal"], deterministic=True)
                obs_, r, terminated, truncated, _ = self.env_evaluate.step(action)
                obs = obs_
                episode_reward += r
                if truncated:
                    break
            evaluate_reward += episode_reward

        return evaluate_reward / times


def make_env(args):
    """ 配置环境 """
    from robopal.assets.robots.diana_med import DianaGrasp

    env = PickAndPlaceEnv(
        robot=DianaGrasp(),
        renderer="viewer",
        is_render=False,
        control_freq=10,
        is_interpolate=False,
        is_pd=False,
        jnt_controller='JNTIMP',
    )
    env = GymWrapper(env)
    env_checker.check_env(env, skip_render_check=True)

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
    setattr(args, 'sigma', 0.2)  # The std of Gaussian noise for exploration

    return env


if __name__ == '__main__':
    args = args()
    env = make_env(args)
    model = HERDDPGModel(
        env=env,
        args=args,
    )
    model.train()

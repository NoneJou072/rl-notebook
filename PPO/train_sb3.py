from stable_baselines3 import A2C, PPO, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback
from gym.wrappers import TimeLimit
from stable_baselines3.common.env_checker import check_env
import gym


def run_train():
    env = gym.make('Walker2d-v2')
    env = TimeLimit(env, max_episode_steps=int(1e3))
    check_env(env)

    # define callback function
    class TensorboardCallback(BaseCallback):
        """
        Custom callback for plotting additional values in tensorboard.
        """

        def __init__(self, log_dir, verbose=0):
            super(TensorboardCallback, self).__init__(verbose)
            self.log_dir = log_dir

        def _on_step(self) -> bool:
            if self.n_calls % 51200 == 0:
                print("Saving new best model")
                self.model.save(self.log_dir + f"/model_saved/PPO/admit_diana_{self.n_calls}")
            return True

    log_dir = "log/"

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log=log_dir,
        device="cuda:0",
    )
    # model = A2C.load("./log/model_saved/admit_diana_51200.zip")
    model.learn(total_timesteps=int(2e6), callback=TensorboardCallback(log_dir=log_dir))
    # model.save("admit_diana")


run_train()

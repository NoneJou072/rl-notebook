from stable_baselines3 import A2C, PPO, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback
from gym.wrappers import TimeLimit
from stable_baselines3.common.env_checker import check_env
import gym

def run_train():
    env = gym.make('CartPole-v1')
    env = TimeLimit(env, max_episode_steps=2000)
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

    model = A2C(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log=log_dir,
        device="cuda:0",
    )
    # model = A2C.load("./log/model_saved/admit_diana_51200.zip")
    model.learn(total_timesteps=int(2e6), callback=TensorboardCallback(log_dir=log_dir))
    # model.save("admit_diana")
    obs = env.reset()
    ep_reward = 0
    for i in range(int(1e6)):
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        ep_reward += reward
        if i % 2000 == 0 or done:
            env.reset()
            print(ep_reward)
            ep_reward = 0


if __name__ == '__main__':
    run_train()

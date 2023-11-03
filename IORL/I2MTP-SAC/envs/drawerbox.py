import numpy as np
from overrides import overrides

from robopal.demos.demo_cube_drawer import DrawerCubeEnv
import robopal.commons.transform as trans
from robopal.commons.gym_wrapper import GoalEnvWrapper


class DrawerBox(DrawerCubeEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = 'DrawerBox-v1'

        self.obs_dim = (32,)
        self.goal_dim = (9,)
        self.action_dim = (4,)

        self.max_episode_steps = 50

    def _get_obs(self) -> dict:
        """ The observation space is 16-dimensional, with the first 3 dimensions corresponding to the position
        of the block, the next 3 dimensions corresponding to the position of the goal, the next 3 dimensions
        corresponding to the position of the gripper, the next 3 dimensions corresponding to the vector
        between the block and the gripper, and the last dimension corresponding to the current gripper opening.
        """
        obs = np.zeros(self.obs_dim)
        dt = self.nsubsteps * self.mj_model.opt.timestep

        obs[0:3] = (  # gripper position in global coordinates
            end_pos := self.get_site_pos('0_grip_site')
        )
        obs[3:6] = (  # handle position in global coordinates
            handle_pos := self.get_site_pos('drawer')
        )
        obs[6:9] = (  # block position in global coordinates
            block_pos := self.get_body_pos('green_block')
        )
        obs[9:12] = end_pos - handle_pos  # distance between the handle and the end
        obs[12:15] = end_pos - block_pos  # distance between the block and the end
        obs[15:18] = trans.mat_2_euler(self.get_body_rotm('cupboard'))
        obs[18:21] = (  # gripper linear velocity
            end_vel := self.get_site_xvelp('0_grip_site') * dt
        )
        # velocity with respect to the gripper
        handle_velp = self.get_site_xvelp('drawer') * dt
        obs[21:24] = (  # velocity with respect to the gripper
            handle_velp - end_vel
        )
        block_velp = self.get_body_xvelp('green_block') * dt
        obs[24:27] = (  # velocity with respect to the gripper
            block_velp - end_vel
        )
        obs[27:30] = self.get_body_xvelr('green_block') * dt
        obs[30] = self.mj_data.joint('0_r_finger_joint').qpos[0]
        obs[31] = self.mj_data.joint('0_r_finger_joint').qvel[0] * dt

        return {
            'observation': obs.copy(),
            'achieved_goal': self._get_achieved_goal(),
            'desired_goal': self._get_desired_goal()
        }

    def _get_achieved_goal(self):
        achieved_goal = np.concatenate([
            self.get_site_pos('0_grip_site'),
            self.get_site_pos('drawer'),
            block_pos := self.get_body_pos('green_block')
        ], axis=0)
        return achieved_goal.copy()

    def _get_desired_goal(self):
        desired_goal = np.concatenate([
            self.get_site_pos('drawer') if self._is_success(
                self.get_site_pos('drawer'), self.get_site_pos('drawer_goal')
            ) == -1 else self.get_body_pos('green_block'),
            self.get_site_pos('drawer_goal'),
            self.get_site_pos('cube_goal'),
        ], axis=0)
        return desired_goal.copy()

    def _get_info(self) -> dict:
        return {
            'is_drawer_success': self._is_success(self.get_site_pos('drawer'), self.get_site_pos('drawer_goal')),
            'is_place_success': self._is_success(self.get_body_pos('green_block'), self.get_site_pos('cube_goal'))
        }

    @overrides
    def reset(self, seed=None):
        super().reset()
        self._timestep = 0
        # set new goal
        self.goal_pos = self.get_site_pos('cube_goal')

        if self.TASK_FLAG == 0:
            pass
        elif self.TASK_FLAG == 1:
            self.mj_data.joint('drawer:joint').qpos[0] = 0.12

        obs = self._get_obs()
        info = self._get_info()

        if self.render_mode == 'human':
            self.render()

        return obs, info


if __name__ == '__main__':
    env = DrawerBox(is_render=True)
    env = GoalEnvWrapper(env)
    env.reset()
    for timestep in range(int(1e5)):
        env.render()
        env.step(env.action_space.sample())
        if timestep % env.max_episode_steps == 0:
            env.reset()
    env.close()

""" Rllab implementation with a HACK. See comment in GymEnv.__init__(). """
import gym
import gym.wrappers
import gym.envs
import gym.spaces
import traceback
import logging

import numpy as np

try:
    from gym import logger as monitor_logger
    monitor_logger.setLevel(logging.WARNING)
except Exception as e:
    traceback.print_exc()

import os
from rllab.envs.base import Env, Step
from rllab.envs.env_spec import EnvSpec
from rllab.core.serializable import Serializable
from rllab.misc import logger
from rllab.spaces.box import Box

from .env_utils import convert_gym_space, Dict


class CappedCubicVideoSchedule(object):
    # Copied from gym, since this method is frequently moved around
    def __call__(self, count):
        if count < 1000:
            return int(round(count ** (1. / 3))) ** 3 == count
        else:
            return count % 1000 == 0


class FixedIntervalVideoSchedule(object):
    def __init__(self, interval):
        self.interval = interval

    def __call__(self, count):
        return count % self.interval == 0


class NoVideoSchedule(object):
    def __call__(self, count):
        return False


class GymEnv(Env, Serializable):
    def __init__(self, env_name, record_video=False, video_schedule=None, log_dir=None, record_log=False,
                 force_reset=True):
        super(GymEnv, self).__init__()
        if log_dir is None:
            if logger.get_snapshot_dir() is None:
                logger.log("Warning: skipping Gym environment monitoring since snapshot_dir not configured.")
            else:
                log_dir = os.path.join(logger.get_snapshot_dir(), "gym_log")
        Serializable.quick_init(self, locals())

        env = gym.envs.make(env_name)

        # HACK: Gets rid of the TimeLimit wrapper that sets 'done = True' when
        # the time limit specified for each environment has been passed and
        # therefore the environment is not Markovian (terminal condition depends
        # on time rather than state).
        env = env.env

        self.env = env
        self.env_id = env.spec.id

        assert not (not record_log and record_video)

        if log_dir is None or record_log is False:
            self.monitoring = False
        else:
            if not record_video:
                video_schedule = NoVideoSchedule()
            else:
                if video_schedule is None:
                    video_schedule = CappedCubicVideoSchedule()
            self.env = gym.wrappers.Monitor(self.env, log_dir, video_callable=video_schedule, force=True)
            self.monitoring = True

        self._observation_space = convert_gym_space(env.observation_space)
        logger.log("observation space: {}".format(self._observation_space))
        self._action_space = convert_gym_space(env.action_space)
        logger.log("action space: {}".format(self._action_space))
        self._horizon = env.spec.tags['wrapper_config.TimeLimit.max_episode_steps']
        self._log_dir = log_dir
        self._force_reset = force_reset

        self._obs_space = None
        self._current_obs_dim = None

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def horizon(self):
        return self._horizon

    def reset(self):
        if self._force_reset and self.monitoring:
            # from gym.wrappers.monitoring import Monitor
            from gym.wrappers.monitor import Monitor
            assert isinstance(self.env, Monitor)
            recorder = self.env.stats_recorder
            if recorder is not None:
                recorder.done = True
        return self.env.reset()

    def step(self, action):
        next_obs, reward, done, info = self.env.step(action)
        return Step(next_obs, reward, done, **info)

    def render(self, mode='human', close=False):
        # return self.env._render(mode, close)
        return self.env.render(mode)
        # self.env.render()

    def terminate(self):
        if self.monitoring:
            # self.env._close()
            self.env.close()
            # self.env.env.close()        # this line is added
            if self._log_dir is not None:
                print("""
    ***************************

    Training finished!

    ***************************
                """)


class GoalEnv(GymEnv):
    """
    A goal-based environment. It functions just as any regular rllab environment but it
    imposes a required structure on the observation_space. More concretely, the observation
    space is required to contain at least three elements, namely `observation`, `desired_goal`, and
    `achieved_goal`. Here, `desired_goal` specifies the goal that the agent should attempt to achieve.
    `achieved_goal` is the goal that it currently achieved instead. `observation` contains the
    actual observations of the environment as per usual.
    """
    def reset(self):
        # Enforce that each GoalEnv uses a Goal-compatible observation space.
        if not isinstance(self.observation_space, Dict):
            raise Exception('GoalEnv requires an observation space of type gym.spaces.Dict')
        result = super(GoalEnv, self).reset()
        for key in ['observation', 'achieved_goal', 'desired_goal']:
            if key not in result:
                raise Exception('GoalEnv requires the "{}" key to be part of the observation dictionary.'.format(key))
        return result

    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        Compute the step reward. This externalizes the reward function and makes
        it dependent on an a desired goal and the one that was achieved. If you wish to include
        additional rewards that are independent of the goal, you can include the necessary values
        to derive it in info and compute it accordingly.
        :param achieved_goal: (`object`) the goal that was achieved during execution
        :param desired_goal: (`object`) the desired goal that we asked the agent to attempt to achieve
        :param info: (`dict`) an info dictionary with additional information
        :return: (`float`) The reward that corresponds to the provided achieved goal w.r.t. to
                the desired goal. Note that the following should always hold true:

                ob, reward, done, info = env.step()
                assert reward == env.compute_reward(ob['achieved_goal'], ob['goal'], info)
        """
        return self.env.compute_reward(achieved_goal, desired_goal, info)

    def goal_env_spec(self, dict_keys=('observation', 'desired_goal')):
        """
        convert selected keys of a Dict observation space to a Box space
        and return the corresponding env_spec.
        :param dict_keys: (`tuple`) desired keys that you would like to use.
        :return: (`EnvSpec`) converted object
        """
        assert isinstance(self.observation_space, Dict)
        obs_dim = np.sum([self.observation_space.spaces[key].flat_dim for key in dict_keys])
        if self._obs_space is None or self._current_obs_dim != obs_dim:
            self._obs_space = Box(-np.inf, np.inf, shape=(obs_dim,))
            self._current_obs_dim = obs_dim

        return EnvSpec(
            observation_space=self._obs_space,
            action_space=self._action_space
        )


class FlattenDictWrapper(GoalEnv, Serializable):
    """
    Flattens selected keys of a Dict observation space into an array.
    Wrap a GymEnv object with a Dict observation space into a flatten observation space env.
    """
    def __init__(self, env_name, dict_keys):
        Serializable.quick_init(self, locals())
        super(FlattenDictWrapper, self).__init__(env_name)
        self.dict_keys = dict_keys

        # Figure out observation_space dimension.
        size = 0
        for key in dict_keys:
            shape = self.env.observation_space.spaces[key].shape
            size += np.prod(shape)
        self._observation_space = Box(-np.inf, np.inf, shape=(size,))

    def observation(self, observation):
        assert isinstance(observation, dict)
        obs = []
        for key in self.dict_keys:
            obs.append(observation[key].ravel())
        return np.concatenate(obs)

    def step(self, action):
        next_obs, reward, done, info = self.env.step(action)
        next_obs = self.observation(next_obs)
        return Step(next_obs, reward, done, **info)

    def reset(self):
        if self._force_reset and self.monitoring:
            # from gym.wrappers.monitoring import Monitor
            from gym.wrappers.monitor import Monitor
            assert isinstance(self.env, Monitor)
            recorder = self.env.stats_recorder
            if recorder is not None:
                recorder.done = True
        observation = self.env.reset()
        return self.observation(observation)

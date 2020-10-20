import numpy as np
import copy

from rllab.core.serializable import Serializable
from .simple_replay_buffer import SimpleReplayBuffer


class HindsightReplayBuffer(SimpleReplayBuffer, Serializable):
    def __init__(self, env, max_replay_buffer_size,
                 n_sampled_goals=4, goal_strategy='future'):
        """
        hindsight replay buffer initialization
        :param env (`GoalEnv`): rllab multi-goal environment object.
        :param max_replay_buffer_size (`int`): replay buffer capacity
        :param n_sampled_goals: the number of goals to be substituted per transition
        :param goal_strategy: strategy for goal sampling
        """
        env_spec = env.goal_env_spec()
        max_replay_buffer_size = int(max_replay_buffer_size)
        Serializable.quick_init(self, locals())
        SimpleReplayBuffer.__init__(self, env_spec, max_replay_buffer_size)

        self._env = env
        self._n_sampled_goals = n_sampled_goals
        self._goal_strategy = goal_strategy

    def _sample_strategy_goal(self, episode, start_idx, strategy='future'):
        """
        sample a hindsight goal using strategy
        :param episode: an episode whose observations are of type Dict
        :param start_idx: current time-step t
        :param strategy: goal sample strategy
        :return: a pseudo goal to be substituted for the actual desired goal using strategy
        """
        if strategy == 'future':
            transition_idx = np.random.choice(np.arange(start_idx + 1, len(episode)))
            transition = episode[transition_idx]
        elif strategy == 'final':
            transition = episode[-1]
        else:
            raise NotImplementedError

        goal = transition[0]['achieved_goal']
        # transition has structure (o,a,r,o2,d)

        return goal

    def add_hindsight_episode(self, episode):
        """
        Add an episode to hindsight replay buffer. We implement HER here.
        :param episode: [(o,a,r,o2,d),(),...()] where o is of type Dict
        :return: None
        """
        for t, transition in enumerate(episode):
            obs, action, reward, next_obs, done = transition
            aug_obs = np.concatenate((obs['observation'], obs['desired_goal']))
            next_aug_obs = np.concatenate((next_obs['observation'], next_obs['desired_goal']))
            self.add_sample(aug_obs, action, reward, done, next_aug_obs)

            # HER
            if t == len(episode) - 1:
                strategy = 'final'
            else:
                strategy = self._goal_strategy

            sampled_achieved_goals = [
                self._sample_strategy_goal(episode=episode, start_idx=t, strategy=strategy)
                for _ in range(self._n_sampled_goals)
            ]

            for goal in sampled_achieved_goals:
                obs, action, reward, next_obs, done = copy.deepcopy(transition)
                obs['desired_goal'] = goal
                next_obs['desired_goal'] = goal
                aug_obs = np.concatenate((obs['observation'], obs['desired_goal']))
                next_aug_obs = np.concatenate((next_obs['observation'], next_obs['desired_goal']))
                reward = self._env.compute_reward(next_obs['achieved_goal'],
                                                  next_obs['desired_goal'],
                                                  info=None)

                self.add_sample(aug_obs, action, reward, done, next_aug_obs)

        self.terminate_episode()

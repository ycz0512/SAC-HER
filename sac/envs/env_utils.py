
import collections
import gym.spaces
import numpy as np

from rllab.spaces.box import Box
from rllab.spaces.discrete import Discrete
from rllab.spaces.product import Product
from rllab.envs.normalized_env import normalize


def convert_gym_space(space):
    """
    Convert a gym.space to an rllab.space
    :param space: (obj:`gym.Space`) The Space object to convert
    :return: converted rllab.Space object
    """
    if isinstance(space, gym.spaces.Box):
        return Box(low=space.low, high=space.high)
    elif isinstance(space, gym.spaces.Discrete):
        return Discrete(n=space.n)
    elif isinstance(space, gym.spaces.Tuple):
        return Product([convert_gym_space(x) for x in space.spaces])
    elif isinstance(space, gym.spaces.Dict):
        return Dict(space.spaces)
    else:
        raise TypeError


class Dict(gym.spaces.Dict):
    """
    rllab Dict Space.
    A dictionary of simpler spaces, e.g. Discrete, Box.
    self.spaces is an OrderedDict which contains keys and corresponding simpler spaces from rllab

    Example usage:
        self.observation_space = Dict({"position": spaces.Discrete(2),
                                       "velocity": spaces.Discrete(3)})
    """

    def __init__(self, spaces=None):
        """
        :param spaces: OrderedDict from gym.spaces.Dict,
                       containing keys and corresponding simpler spaces.
        """
        super(Dict, self).__init__(spaces)
        self.spaces = (collections.OrderedDict([
            (k, convert_gym_space(s)) for k, s in self.spaces.items()
        ]))

    @property
    def flat_dim(self):
        """Return the length of the flattened vector of the space."""
        return sum([space.flat_dim for _, space in self.spaces.items()])

    def flat_dim_with_keys(self, keys):
        """Return a flat dimension of the spaces specified by the keys.

        Returns:
            sum (int)

        """
        return sum([self.spaces[key].flat_dim for key in keys])

    def flatten(self, x):
        """Return an observation of x with collapsed values.

        Args:
            x (:obj:`Iterable`): The object to flatten.

        Returns:
            Dict: A Dict where each value is collapsed into a single dimension.
                  Keys are unchanged.

        """
        return np.concatenate(
            [space.flatten(x[key]) for key, space in self.spaces.items()],
            axis=-1,
        )

    def unflatten(self, x):
        """Return an unflattened observation x.

        Args:
            x (:obj:`Iterable`): The object to unflatten.

        Returns:
            collections.OrderedDict

        """
        dims = np.array([s.flat_dim for s in self.spaces.values()])
        flat_x = np.split(x, np.cumsum(dims)[:-1])
        return collections.OrderedDict([
            (key, self.spaces[key].unflatten(xi))
            for key, xi in zip(self.spaces.keys(), flat_x)
        ])

    def flatten_n(self, xs):
        """Return flattened observations xs.

        Args:
            xs (:obj:`Iterable`): The object to reshape and flatten

        Returns:
            np.ndarray: An array of xs in a shape inferred by the size of
                its first element.

        """
        return np.array([self.flatten(x) for x in xs])

    def unflatten_n(self, xs):
        """Return unflattened observations xs.

        Args:
            xs (:obj:`Iterable`): The object to reshape and unflatten

        Returns:
            List[OrderedDict]

        """
        return [self.unflatten(x) for x in xs]

    def flatten_with_keys(self, x, keys):
        """Return flattened obs of spaces specified by the keys using x.

        Returns:
            list

        """
        return np.concatenate(
            [
                space.flatten(x[key])
                for key, space in self.spaces.items() if key in keys
            ],
            axis=-1,
        )

    def unflatten_with_keys(self, x, keys):
        """Return an unflattened observation.

        This is the inverse of `flatten_with_keys`.

        Returns:
            collections.OrderedDict

        """
        dims = np.array([
            space.flat_dim for key, space in self.spaces.items() if key in keys
        ])
        flat_x = np.split(x, np.cumsum(dims)[:-1])
        return collections.OrderedDict([
            (key, space.unflatten(xi))
            for (key, space), xi in zip(self.spaces.items(), flat_x)
            if key in keys
        ])


class normalize_goal_env(normalize):
    def __init__(self, env):
        """
        normalize for multi-goal env
        :param env: ('GoalEnv') gym_env.GoalEnv object
        """
        super(normalize_goal_env, self).__init__(env=env)
        self._wrapped_env = env     # wrapped_env is goal_env (gym_env.GoalEnv object)

    def compute_reward(self, achieved_goal, desired_goal, info):
        return self._wrapped_env.compute_reward(achieved_goal, desired_goal, info)

    def goal_env_spec(self, dict_keys=('observation', 'desired_goal')):
        return self._wrapped_env.goal_env_spec(dict_keys)

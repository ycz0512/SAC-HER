import datetime
import dateutil.tz
import os
import numpy as np


def timestamp():
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    return now.strftime('%Y-%m-%d-%H-%M-%S-%f-%Z')


def flatten_obs(observation, dict_keys=('observation', 'desired_goal')):
    """
    Flattens selected keys of a Dict observation space into an array.
    :param observation: (`Dict`) GoalEnv observation
    :param dict_keys: (`tuple`) desired keys that you would like to use.
    :return: (`np.array`) flattened observation with selected keys
    """
    assert isinstance(observation, dict)
    obs = []
    for key in dict_keys:
        obs.append(observation[key].ravel())
    return np.concatenate(obs)


PROJECT_PATH = os.path.dirname(
    os.path.realpath(os.path.join(__file__, '..', '..')))

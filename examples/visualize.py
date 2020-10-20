import argparse

import joblib
import tensorflow as tf

from rllab.sampler.utils import rollout
from sac.misc.sampler import rollouts

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='Path to the snapshot file.')
    parser.add_argument('--max-path-length', '-l', type=int, default=20)
    parser.add_argument('--speedup', '-s', type=float, default=1)
    parser.add_argument('--deterministic', '-d', dest='deterministic',
                        action='store_true')
    parser.add_argument('--no-deterministic', '-nd', dest='deterministic',
                        action='store_false')
    parser.set_defaults(deterministic=True)

    args = parser.parse_args()

    with tf.Session() as sess:
        data = joblib.load(args.file)
        if 'algo' in data.keys():
            policy = data['algo'].policy
            env = data['algo'].env
        else:
            policy = data['policy']
            env = data['env']

        with policy.deterministic(args.deterministic):
            while True:
                _ = rollouts(env, policy, path_length=args.max_path_length,
                             n_paths=1, render=True, is_goal_env=True,
                             speedup=args.speedup)

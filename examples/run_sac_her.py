import argparse

from rllab.misc.instrument import VariantGenerator

from sac.algos.sac_her import SAC_HER
from sac.envs.gym_env import GoalEnv
from sac.policies.gmm import GMMPolicy
from sac.replay_buffers.hindsight_replay_buffer import HindsightReplayBuffer
from sac.value_functions import NNQFunction, NNVFunction
from sac.envs.env_utils import normalize_goal_env
from sac.misc.instrument import run_sac_experiment
from sac.misc.utils import timestamp


SHARED_PARAMS = {
    "seed": [1, 2, 3],
    "lr": 3E-4,
    "discount": 0.99,
    "tau": 0.01,
    "K": 4,
    "layer_size": 256,
    "batch_size": 256,
    "max_pool_size": 1E6,
    "n_sampled_goals": 4,
    "goal_strategy": 'future',
    "scale_reward": 3,
    "n_train_repeat": 1,
    "epoch_length": 2500,
    "snapshot_mode": 'gap',
    "snapshot_gap": 5,
    "sync_pkl": True,
}

ENV_PARAMS = {
    'FetchReach': {  # 7 DoF
        'prefix': 'FetchReach',
        'env_name': 'FetchReach-v1',
        'max_path_length': 50,
        'n_epochs': 20,
    },
    'FetchPush': {  # 7 DoF
        'prefix': 'FetchPush',
        'env_name': 'FetchPush-v1',
        'max_path_length': 50,
        'n_epochs': 600,
    },
    'FetchSlide': {  # 7 DoF
        'prefix': 'FetchSlide',
        'env_name': 'FetchSlide-v1',
        'max_path_length': 50,
        'n_epochs': 800,
    },
    'FetchPickAndPlace': {  # 7 DoF
        'prefix': 'FetchPickAndPlace',
        'env_name': 'FetchPickAndPlace-v1',
        'max_path_length': 50,
        'n_epochs': 800,
    },
    'HandReach': {  # 24 DoF
        'prefix': 'HandReach',
        'env_name': 'HandReach-v0',
        'max_path_length': 50,
        'n_epochs': 1000,
    },
    'HandBlock': {  # 24 DoF
        'prefix': 'HandManipulateBlock',
        'env_name': 'HandManipulateBlock-v0',
        'max_path_length': 100,
        'n_epochs': 1500,
    },
    'HandEgg': {  # 24 DoF
        'prefix': 'HandManipulateEgg',
        'env_name': 'HandManipulateEgg-v0',
        'max_path_length': 100,
        'n_epochs': 1500,
    },
    'HandPen': {  # 24 DoF
        'prefix': 'HandManipulatePen',
        'env_name': 'HandManipulatePen-v0',
        'max_path_length': 100,
        'n_epochs': 2000,
    },
}

DEFAULT_ENV = 'FetchReach'
AVAILABLE_ENVS = list(ENV_PARAMS.keys())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, choices=AVAILABLE_ENVS, default=DEFAULT_ENV)
    parser.add_argument('--exp_name', type=str, default=timestamp())
    parser.add_argument('--log_dir', type=str, default=None)
    arg_parser = parser.parse_args()

    return arg_parser


def get_variants(arg_parser):
    env_params = ENV_PARAMS[arg_parser.env]
    params = SHARED_PARAMS
    params.update(env_params)

    variant_generator = VariantGenerator()
    for key, val in params.items():
        if isinstance(val, list):
            variant_generator.add(key, val)
        else:
            variant_generator.add(key, [val])

    return variant_generator


def run_experiment(variant):
    env = normalize_goal_env(GoalEnv(variant['env_name']))
    env_spec = env.goal_env_spec()

    her_pool = HindsightReplayBuffer(
        env=env,
        max_replay_buffer_size=variant['max_pool_size'],
        n_sampled_goals=variant["n_sampled_goals"],
        goal_strategy=variant["goal_strategy"],
    )

    M = variant['layer_size']
    qf = NNQFunction(
        env_spec=env_spec,
        hidden_layer_sizes=[M, M],
    )

    vf = NNVFunction(
        env_spec=env_spec,
        hidden_layer_sizes=[M, M],
    )

    policy = GMMPolicy(
        env_spec=env_spec,
        K=variant['K'],
        hidden_layer_sizes=[M, M],
        qf=qf,
        reg=0.001,
    )

    base_kwargs = dict(
        min_pool_size=variant['max_path_length'],
        epoch_length=variant['epoch_length'],
        n_epochs=variant['n_epochs'],
        max_path_length=variant['max_path_length'],
        batch_size=variant['batch_size'],
        n_train_repeat=variant['n_train_repeat'],
        eval_render=False,
        eval_n_episodes=10,
        eval_deterministic=True,
    )

    algorithm = SAC_HER(
        base_kwargs=base_kwargs,
        env=env,
        policy=policy,
        pool=her_pool,
        qf=qf,
        vf=vf,

        lr=variant['lr'],
        scale_reward=variant['scale_reward'],
        discount=variant['discount'],
        tau=variant['tau'],

        save_full_state=False,
    )

    algorithm.train()


def launch_experiments(variant_generator):
    variants = variant_generator.variants()

    for i, variant in enumerate(variants):
        print('Launching {} experiments.'.format(len(variants)))
        run_sac_experiment(
            run_experiment,
            mode='local',
            variant=variant,
            exp_prefix=variant['prefix'] + '/' + args.exp_name,
            exp_name=variant['prefix'] + '-' + args.exp_name + '-' + str(i).zfill(2),
            n_parallel=1,
            seed=variant['seed'],
            terminate_machine=True,
            log_dir=args.log_dir,
            snapshot_mode=variant['snapshot_mode'],
            snapshot_gap=variant['snapshot_gap'],
            sync_s3_pkl=variant['sync_pkl'],
        )


if __name__ == '__main__':
    args = parse_args()
    vg = get_variants(args)
    launch_experiments(vg)

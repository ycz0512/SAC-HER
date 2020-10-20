import gtimer as gt
import numpy as np

from rllab.misc import logger
from rllab.misc.overrides import overrides

from .sac import SAC
from sac.misc.utils import flatten_obs
from sac.misc.sampler import rollouts


class SAC_HER(SAC):
    """
    SAC with Hindsight Experience Replay.

    Example:
        env = normalize_goal_env(GoalEnv("FetchReach-v1"))
        env_spec = env.goal_env_spec()

        her_pool = HindsightReplayBuffer(env, 1E6)

        M = 128
        qf = NNQFunction(env_spec=env_spec, hidden_layer_sizes=[M, M])

        vf = NNVFunction(env_spec=env_spec, hidden_layer_sizes=[M, M])

        policy = GMMPolicy(
            env_spec=env_spec,
            K=4,
            hidden_layer_sizes=[M, M],
            qf=qf,
            reg=0.001,
        )

        base_kwargs = dict(
            min_pool_size=1000,
            epoch_length=1000,
            n_epochs=4,
            max_path_length=1000,
            batch_size=128,
            n_train_repeat=4,
            eval_render=True,
            eval_n_episodes=4,
            eval_deterministic=True,
        )

        algorithm = SAC_HER(
            base_kwargs=base_kwargs,
            env=env,
            policy=policy,
            pool=her_pool,
            qf=qf,
            vf=vf,
        )

        algorithm.train()

    Note:
        env (`gym_env.GoalEnv`): rllab goal-based environment object.
        policy (`rllab.NNPolicy`): A policy function approximator with augmented input s||g.
        qf (`ValueFunction`): Q-function approximator with augmented input s||g.
        vf (`ValueFunction`): Soft value function approximator with augmented input s||g.
        pool: (`HindsightReplayBuffer`): HER replay buffer object.
    """

    @overrides
    def _train(self, env, policy, pool):
        self._init_training(env, policy, pool)

        with self._sess.as_default():
            observation = env.reset()
            policy.reset()

            path_length = 0
            path_return = 0
            last_path_return = 0
            max_path_return = -np.inf
            n_episodes = 0
            trajectory = []
            gt.rename_root('RLAlgorithm')
            gt.reset()
            gt.set_def_unique(False)

            for epoch in gt.timed_for(range(self._n_epochs + 1),
                                      save_itrs=True):
                logger.push_prefix('Epoch #%d | ' % epoch)

                for t in range(self._epoch_length):
                    iteration = t + epoch * self._epoch_length

                    # TODO: flatten the observation in the right place
                    action, _ = policy.get_action(flatten_obs(observation))
                    next_ob, reward, terminal, info = env.step(action)

                    assert reward == env.compute_reward(next_ob['achieved_goal'],
                                                        next_ob['desired_goal'],
                                                        info)

                    path_length += 1
                    path_return += reward

                    trajectory.append(
                        (observation,
                         action,
                         reward,
                         next_ob,
                         terminal)
                    )

                    if terminal or path_length >= self._max_path_length:
                        self._pool.add_hindsight_episode(
                            episode=trajectory
                        )

                        observation = env.reset()
                        policy.reset()
                        path_length = 0
                        max_path_return = max(max_path_return, path_return)
                        last_path_return = path_return

                        path_return = 0
                        n_episodes += 1
                        trajectory = []

                    else:
                        observation = next_ob

                    gt.stamp('sample')

                    if self._pool.size >= self._min_pool_size:
                        for i in range(self._n_train_repeat):
                            batch = self._pool.random_batch(self._batch_size)
                            self._do_training(iteration, batch)

                    gt.stamp('train')

                self._evaluate(epoch)

                params = self.get_snapshot(epoch)
                logger.save_itr_params(epoch, params)
                times_itrs = gt.get_times().stamps.itrs

                eval_time = times_itrs['eval'][-1] if epoch > 1 else 0
                total_time = gt.get_times().total
                logger.record_tabular('time-train', times_itrs['train'][-1])
                logger.record_tabular('time-eval', eval_time)
                logger.record_tabular('time-sample', times_itrs['sample'][-1])
                logger.record_tabular('time-total', total_time)
                logger.record_tabular('epoch', epoch)
                logger.record_tabular('episodes', n_episodes)
                logger.record_tabular('steps', iteration)       # also record total steps
                logger.record_tabular('max-path-return', max_path_return)
                logger.record_tabular('last-path-return', last_path_return)
                logger.record_tabular('pool-size', self._pool.size)

                logger.dump_tabular(with_prefix=False)
                logger.pop_prefix()

                gt.stamp('eval')

            env.terminate()

    def _evaluate(self, epoch):
        if self._eval_n_episodes < 1:
            return

        with self._policy.deterministic(self._eval_deterministic):
            paths = rollouts(self._eval_env, self._policy,
                             self._max_path_length, self._eval_n_episodes,
                             render=False, is_goal_env=True)

        total_returns = [path['rewards'].sum() for path in paths]
        episode_lengths = [len(p['rewards']) for p in paths]
        success = [True in path['dones'] for path in paths]

        logger.record_tabular('return-average', np.mean(total_returns))
        logger.record_tabular('return-min', np.min(total_returns))
        logger.record_tabular('return-max', np.max(total_returns))
        logger.record_tabular('return-std', np.std(total_returns))
        logger.record_tabular('episode-length-avg', np.mean(episode_lengths))
        logger.record_tabular('episode-length-min', np.min(episode_lengths))
        logger.record_tabular('episode-length-max', np.max(episode_lengths))
        logger.record_tabular('episode-length-std', np.std(episode_lengths))
        logger.record_tabular('test-success-rate', np.mean(success))

        self._eval_env.log_diagnostics(paths)
        if self._eval_render:
            self._eval_env.render(paths)

        batch = self._pool.random_batch(self._batch_size)
        self.log_diagnostics(batch)

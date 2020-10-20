import gtimer as gt
import numpy as np

from rllab.misc import logger
from rllab.misc.overrides import overrides

from .sac import SAC
from sac.misc.utils import flatten_obs
from sac.misc.sampler import rollouts


class GoalSAC(SAC):
    """
    SAC with Goal-based environment.

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
            observation = flatten_obs(observation)
            policy.reset()

            path_length = 0
            path_return = 0
            last_path_return = 0
            max_path_return = -np.inf
            n_episodes = 0
            gt.rename_root('RLAlgorithm')
            gt.reset()
            gt.set_def_unique(False)

            for epoch in gt.timed_for(range(self._n_epochs + 1),
                                      save_itrs=True):
                logger.push_prefix('Epoch #%d | ' % epoch)

                for t in range(self._epoch_length):
                    iteration = t + epoch * self._epoch_length

                    action, _ = policy.get_action(observation)

                    next_ob, reward, terminal, info = env.step(action)
                    next_ob = flatten_obs(next_ob)

                    path_length += 1
                    path_return += reward

                    self._pool.add_sample(
                        observation,
                        action,
                        reward,
                        terminal,
                        next_ob,
                    )

                    if terminal or path_length >= self._max_path_length:
                        observation = env.reset()
                        observation = flatten_obs(observation)
                        policy.reset()
                        path_length = 0
                        max_path_return = max(max_path_return, path_return)
                        last_path_return = path_return

                        path_return = 0
                        n_episodes += 1

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

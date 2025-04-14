from typing import Optional, TypeVar

from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.type_aliases import (
    MaybeCallback,
    RolloutReturn,
    GymEnv,
    TrainFrequencyUnit,
)
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise
from stable_baselines3.common.buffers import ReplayBuffer

SelfAsyncOffPolicyAlgorithm = TypeVar(
    "SelfAsyncOffPolicyAlgorithm", bound="AsyncOffPolicyAlgorithm"
)


class AsyncOffPolicyAlgorithm(OffPolicyAlgorithm):
    """Off policy algorithm baseclass which makes use of SubprocVecEnv to do environment steps and model optimization in parallel"""

    def learn(
        self: SelfAsyncOffPolicyAlgorithm,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "run",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfAsyncOffPolicyAlgorithm:
        """
        Return a trained model.

        Based on OffPolicyAlgorithm.learn, with the difference that collect_rollouts has
        been split into async_collect_rollouts and await_collect_rollouts

        :param total_timesteps: The total number of samples (env steps) to train on
        :param callback: callback(s) called at every step with state of the algorithm.
        :param log_interval: The number of timesteps before logging.
        :param tb_log_name: the name of the run for TensorBoard logging
        :param eval_env: Environment that will be used to evaluate the agent. Caution, this parameter
            is deprecated and will be removed in the future. Please use ``EvalCallback`` instead.
        :param eval_freq: Evaluate the agent every ``eval_freq`` timesteps (this may vary a little).
            Caution, this parameter is deprecated and will be removed in the future.
            Please use `EvalCallback` or a custom Callback instead.
        :param n_eval_episodes: Number of episode to evaluate the agent
        :param eval_log_path: Path to a folder where the evaluations will be saved
        :param reset_num_timesteps: whether or not to reset the current timestep number (used in logging)
        :param progress_bar: Display a progress bar using tqdm and rich.
        :return: the trained model
        """

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            eval_env,
            callback,
            eval_freq,
            n_eval_episodes,
            eval_log_path,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        assert self.train_freq.frequency == 1, "Should always collect one step."
        assert self.train_freq.unit == TrainFrequencyUnit.STEP, (
            "Should always collect one step."
        )

        while self.num_timesteps <= total_timesteps:
            buffer_actions = self.async_collect_rollouts(
                self.env,
                action_noise=self.action_noise,
                callback=callback,
                learning_starts=self.learning_starts,
            )

            if self.replay_buffer.size() > self.learning_starts:
                # Special case when the user passes `gradient_steps=0`
                if self.gradient_steps > 0:
                    self.train(
                        batch_size=self.batch_size, gradient_steps=self.gradient_steps
                    )

            rollout = self.await_collect_rollouts(
                self.env,
                buffer_actions=buffer_actions,
                action_noise=self.action_noise,
                callback=callback,
                replay_buffer=self.replay_buffer,
                log_interval=log_interval,
            )

            if rollout.continue_training is False:
                break

        callback.on_training_end()

        return self

    def async_collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
    ):
        """
        Starts a single async step in the vectorized environment.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param train_freq: How much experience to collect
            by doing rollouts of current policy.
            Either ``TrainFreq(<n>, TrainFrequencyUnit.STEP)``
            or ``TrainFreq(<n>, TrainFrequencyUnit.EPISODE)``
            with ``<n>`` being an integer greater than 0.
        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :return: buffer_actions: Actions to be stored in the replay buffer
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        assert isinstance(env, VecEnv), "You must pass a VecEnv"

        # Vectorize action noise if needed
        if (
            action_noise is not None
            and env.num_envs > 1
            and not isinstance(action_noise, VectorizedActionNoise)
        ):
            action_noise = VectorizedActionNoise(action_noise, env.num_envs)

        if self.use_sde:
            self.actor.reset_noise(env.num_envs)

        callback.on_rollout_start()

        if self.use_sde and self.sde_sample_freq > 0:
            # Sample a new noise matrix
            self.actor.reset_noise(env.num_envs)

        # Select action randomly or according to policy
        actions, buffer_actions = self._sample_action(
            learning_starts, action_noise, env.num_envs
        )

        # Rescale and perform action async
        env.step_async(actions)

        return buffer_actions

    def await_collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        replay_buffer: ReplayBuffer,
        buffer_actions,
        action_noise: Optional[ActionNoise] = None,
        log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        """
        Collect experience from a previously started async step and stores it into a ``ReplayBuffer``.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param replay_buffer:
        :param log_interval: Log data every ``log_interval`` episodes
        :param buffer_actions: Actions to be stored in the replay buffer. Generated in the ``async_collect_rollouts`` function.
        :return:
        """

        # Await results
        new_obs, rewards, dones, infos = env.step_wait()

        self.num_timesteps += env.num_envs

        # Give access to local variables
        callback.update_locals(locals())
        # Only stop training if return value is False, not when it is None.
        if callback.on_step() is False:
            return RolloutReturn(env.num_envs, 0, continue_training=False)

        # Retrieve reward and episode length if using Monitor wrapper
        self._update_info_buffer(infos, dones)

        # Store data in replay buffer (normalized action and unnormalized observation)
        self._store_transition(
            replay_buffer, buffer_actions, new_obs, rewards, dones, infos
        )

        self._update_current_progress_remaining(
            self.num_timesteps, self._total_timesteps
        )

        # For DQN, check if the target network should be updated
        # and update the exploration schedule
        # For SAC/TD3, the update is dones as the same time as the gradient update
        # see https://github.com/hill-a/stable-baselines/issues/900
        self._on_step()

        num_collected_episodes = 0
        for idx, done in enumerate(dones):
            if done:
                # Update stats
                num_collected_episodes += 1
                self._episode_num += 1

                if action_noise is not None:
                    kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                    action_noise.reset(**kwargs)

                # Log training infos
                if log_interval is not None and self._episode_num % log_interval == 0:
                    self._dump_logs()

        callback.on_rollout_end()

        return RolloutReturn(env.num_envs, num_collected_episodes, True)

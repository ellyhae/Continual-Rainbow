from typing import Optional, Tuple, TypeVar

from gym import spaces
import numpy as np

from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.type_aliases import MaybeCallback, TrainFreq, RolloutReturn, GymEnv, TrainFrequencyUnit
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.buffers import ReplayBuffer

SelfAsyncOffPolicyAlgorithm = TypeVar("SelfAsyncOffPolicyAlgorithm", bound="AsyncOffPolicyAlgorithm")

class AsyncOffPolicyAlgorithm(OffPolicyAlgorithm):
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
        assert self.train_freq.unit == TrainFrequencyUnit.STEP, "Should always collect one step."
    
        while self.num_timesteps <= total_timesteps:  #  + self.env.num_envs  adjust limit as the last set of experiences is not used for training
            buffer_actions = self.async_collect_rollouts(
                self.env,
                action_noise=self.action_noise,
                callback=callback,
                learning_starts=self.learning_starts,
            )
    
            if self.replay_buffer.size() > self.learning_starts:
                # Special case when the user passes `gradient_steps=0`
                if self.gradient_steps > 0:
                    self.train(batch_size=self.batch_size, gradient_steps=self.gradient_steps)

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
        if action_noise is not None and env.num_envs > 1 and not isinstance(action_noise, VectorizedActionNoise):
            action_noise = VectorizedActionNoise(action_noise, env.num_envs)

        if self.use_sde:
            self.actor.reset_noise(env.num_envs)

        callback.on_rollout_start()
        
        #if self.use_sde and self.sde_sample_freq > 0:
        #    # Sample a new noise matrix
        #    self.actor.reset_noise(env.num_envs)

        # Select action randomly or according to policy
        actions, buffer_actions = self._sample_action(learning_starts, action_noise, env.num_envs)

        # Rescale and perform action async
        env.step_async(actions)
        
        return buffer_actions
    
    def _sample_action(
        self,
        learning_starts: int,
        action_noise: Optional[ActionNoise] = None,
        n_envs: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Modified to assume exploration rate schedule, and therefore use the model in warmup
        
        Sample an action according to the exploration policy.
        This is either done by sampling the probability distribution of the policy,
        or sampling a random action (from a uniform distribution over the action space)
        or by adding noise to the deterministic output.

        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param n_envs:
        :return: action to take in the environment
            and scaled action that will be stored in the replay buffer.
            The two differs when the action space is not normalized (bounds are not [-1, 1]).
        """
        # Select action according to policy
        unscaled_action, _ = self.predict(self._last_obs, deterministic=False)

        # Rescale the action from [low, high] to [-1, 1]
        if isinstance(self.action_space, spaces.Box):
            scaled_action = self.policy.scale_action(unscaled_action)

            # Add noise to the action (improve exploration)
            if action_noise is not None:
                scaled_action = np.clip(scaled_action + action_noise(), -1, 1)

            # We store the scaled action in the buffer
            buffer_action = scaled_action
            action = self.policy.unscale_action(scaled_action)
        else:
            # Discrete case, no need to normalize or clip
            buffer_action = unscaled_action
            action = buffer_action
        return action, buffer_action
        
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
        self._store_transition(replay_buffer, buffer_actions, new_obs, rewards, dones, infos)

        self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

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
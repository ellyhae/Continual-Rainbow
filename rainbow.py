import warnings
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union, Literal

import numpy as np
import torch as torch
from torch import Tensor
from torch.cuda.amp import GradScaler, autocast
from gym import spaces

from stable_baselines3.common.buffers import BaseBuffer
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import (
    get_linear_fn,
    get_parameters_by_name,
    is_vectorized_observation,
    polyak_update,
)
from stable_baselines3.common.preprocessing import maybe_transpose
from stable_baselines3.common.callbacks import BaseCallback

from async_algorithm import AsyncOffPolicyAlgorithm
from policies import CnnPolicy, RainbowPolicy, MlpPolicy, RainbowNetwork
from torch_layers import FactorizedNoisyLinear
from buffer import PrioritizedReplayBuffer

SelfRainbow = TypeVar("SelfRainbow", bound="Rainbow")


@torch.no_grad()
def reset_noise(net: torch.nn.Module) -> None:
    """sample new weights/biases for all FactorizedNoisyLinear in the network"""
    for m in net.modules():
        if isinstance(m, FactorizedNoisyLinear):
            m.reset_noise()


class NoiseReset(BaseCallback):
    """Automatically sample new weights/biases for all FactorizedNoisyLinear
    in the online q-network before a new transition is collected from the environment"""

    def _on_rollout_start(self) -> None:
        reset_noise(self.model.q_net)

    def _on_step(self) -> bool:
        """This callback never stops the training"""
        return True


class Rainbow(AsyncOffPolicyAlgorithm):
    """
    Rainbow Algorithm

    Paper: http://arxiv.org/abs/1710.02298 http://arxiv.org/abs/2111.10247

    Default hyperparameters and code were mainly adapted from https://github.com/schmidtdominik/Rainbow

    Much of the code structure is based on stable baseline 3's DQN implementation
    """

    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": None,
    }
    # Linear schedule will be defined in `_setup_model()`
    exploration_schedule: Schedule
    q_net: RainbowNetwork
    q_net_target: RainbowNetwork
    policy: RainbowPolicy

    def __init__(
        self,
        policy: Union[str, Type[RainbowPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 0.00025,
        buffer_size: int = int(2**20),
        learning_starts: int = 100_000,
        batch_size: int = 256,
        tau: float = 1.0,
        gamma: float = 0.99,
        gradient_steps: int = 2,
        replay_buffer_class: Optional[Type[BaseBuffer]] = PrioritizedReplayBuffer,
        replay_buffer_kwargs: Dict[str, Any] | None = None,
        optimize_memory_usage: bool = False,
        target_update_interval: int = 32_000,
        exploration_fraction: float = 0.05,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.01,
        double_dqn: bool = True,
        max_grad_norm: float = 10,
        loss_fn: Literal["huber", "mse"] = "huber",
        tensorboard_log: str | None = None,
        policy_kwargs: Dict[str, Any] | None = {"optimizer_kwargs": {"eps": None}},
        verbose: Literal[0, 1, 2] = 0,
        seed: int | None = None,
        use_amp: bool = True,
        device: Union[torch.device, str] = "auto",
        _init_setup_model: bool = True,
    ) -> None:
        """_summary_

        Args:
            policy (Union[str, Type[RainbowPolicy]]): The policy model to use (MlpPolicy, CnnPolicy)
            env (Union[GymEnv, str]): The environment to learn from (if registered in Gym, can be str)
            learning_rate (Union[float, Schedule], optional): The learning rate, it can be a function
                of the current progress remaining (from 1 to 0). Defaults to 0.00025.
            buffer_size (int, optional): size of the replay buffer. Defaults to int(2**20).
            learning_starts (int, optional): how many steps of the model to collect transitions for before learning starts.
                Defaults to 100_000.
            batch_size (int, optional): Minibatch size for each gradient update. Defaults to 256.
            tau (float, optional): the soft update coefficient ("Polyak update", between 0 and 1). Defaults to 1.0, hard update.
            gamma (float, optional): the discount factor for a single step. May be adapted internally to fit n-step TD-target calculation.
                Defaults to 0.99.
            gradient_steps (int, optional): How many gradient steps to do during each rollout. Defaults to 2.
            replay_buffer_class (Optional[Type[BaseBuffer]], optional): Replay buffer class to use.
                If ``PrioritizedReplayBuffer``, then the ``gamma`` of this class is adjusted to use
                the ``n_step`` bootstrapping gamma. Defaults to PrioritizedReplayBuffer.
            replay_buffer_kwargs (Dict[str, Any] | None, optional): Keyword arguments to pass to the replay buffer on creation.
                If ``replay_buffer_class`` is ``PrioritizedReplayBuffer`` then the ``gamma`` and ``use_amp`` arguments
                of this class will be provided to the replay buffer. Defaults to None.
            optimize_memory_usage (bool, optional): Enable a memory efficient variant of the replay buffer
                at a cost of more complexity.
                See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195.
                Note: it depends on the ``replay_buffer_class`` if this parameter has any effect.
                Defaults to False.
            target_update_interval (int, optional): update the target network every ``target_update_interval``
                timesteps. Note: when using a vectorized environment a single environment step + optimization
                is counted as ``n_envs`` timesteps. Defaults to 32_000.
            exploration_fraction (float, optional): fraction of entire training period over which the exploration rate is reduced. Defaults to 0.05.
            exploration_initial_eps (float, optional): initial value of random action probability. Defaults to 1.0.
            exploration_final_eps (float, optional): final value of random action probability. Defaults to 0.01.
            double_dqn (bool, optional):  Whether or not to use a target network for TD-target calculation. Defaults to True.
            max_grad_norm (float, optional): The maximum value for the gradient clipping. Defaults to 10.
            loss_fn (Literal["huber", "mse"], optional): Loss function used in the optimization. Defaults to "huber".
            tensorboard_log (str | None, optional): the log location for tensorboard (if None, no logging). Defaults to None.
            policy_kwargs (Dict[str, Any] | None, optional): additional arguments to be passed to the policy on creation.
                If policy_kwargs.optimizer_kwargs.eps is set to None, then it is replaced by 0.005 / batch_size.
                Defaults to {"optimizer_kwargs": {"eps": None}}.
            verbose (Literal[0, 1, 2], optional): Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
                debug messages. Defaults to 0.
            seed (int | None, optional): Seed for the pseudo random generators. Defaults to None.
            use_amp (bool, optional): Whether or not to use Pytorch Automatic Mixed Precision. Defaults to True.
            device (Union[torch.device, str], optional): Device (cpu, cuda, ...) on which the code should be run.
                Setting it to auto, the code will be run on the GPU if possible.. Defaults to "auto".
            _init_setup_model (bool, optional): Whether or not to build the network at the creation of the instance. Defaults to True.
        """
        self.prioritized_er = replay_buffer_class == PrioritizedReplayBuffer

        if self.prioritized_er:
            if replay_buffer_kwargs is None:
                replay_buffer_kwargs = {}

            replay_buffer_kwargs["gamma"] = gamma
            replay_buffer_kwargs["use_amp"] = use_amp

        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.exploration_fraction = exploration_fraction

        if policy_kwargs is not None:
            # if the eps parameter of the optimizer is set to None, then it is replaced by a computed value
            # Note: if the eps parameter is not in the optimizer kwargs, then the default of the optimizer is used
            if policy_kwargs.get("optimizer_kwargs", {}).get("eps", ...) is None:
                policy_kwargs["optimizer_kwargs"]["eps"] = 0.005 / batch_size

            # if noisy linear layers are enabled, then the epsilon-greedy schedule needs to be adjusted
            if policy_kwargs.get("noisy_linear", False):
                self.exploration_initial_eps = 0.002
                self.exploration_final_eps = 0.0
                self.exploration_fraction = 0.002
                print("Updated exploration parameters")

        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            # AsyncOffPolicyAlgorithm currently only supports 1 step in parllel
            train_freq=(1, "step"),
            gradient_steps=gradient_steps,
            action_noise=None,  # No action noise
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            sde_support=False,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(spaces.Discrete,),
            support_multi_env=True,
        )

        self.exploration_rate = self.exploration_initial_eps

        self.target_update_interval = target_update_interval
        # For updating the target network with multiple envs:
        assert self.target_update_interval % self.n_envs == 0, (
            "target would not be synced since the main loop iterates in steps of parallel_envs"
        )
        self.max_grad_norm = max_grad_norm

        self.double_dqn = double_dqn

        # if PER is being used, then a weighted average is computed in the train method
        # for that the loss needs to not be reduced
        loss_fn_cls = torch.nn.MSELoss if loss_fn == "mse" else torch.nn.SmoothL1Loss
        self.loss_fn = loss_fn_cls(
            reduction=("none" if self.prioritized_er else "mean")
        )

        # define gradient scaler, which makes sure that optimization
        # works as expected even with lower accuracy (automatic mixed precision)
        self.use_amp = use_amp
        self.scaler = GradScaler(enabled=self.use_amp)

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        """Create networks, buffer and optimizers.

        Slightly modified version of stable baseline's DQN._on_step, but without
        correction of the target_update_interval. That is handled in _on_step"""

        # initialize policy and replay buffer
        super()._setup_model()

        self._create_aliases()
        self.noisy_dqn = self.policy.noisy_linear

        # Copy running stats, see GH issue #996
        self.batch_norm_stats = get_parameters_by_name(self.q_net, ["running_"])
        self.batch_norm_stats_target = get_parameters_by_name(
            self.q_net_target, ["running_"]
        )
        self.exploration_schedule = get_linear_fn(
            self.exploration_initial_eps,
            self.exploration_final_eps,
            self.exploration_fraction,
        )
        # Account for multiple environments
        # each call to step() corresponds to n_envs transitions
        if self.n_envs > 1:
            if self.n_envs > self.target_update_interval:
                warnings.warn(
                    "The number of environments used is greater than the target network "
                    f"update interval ({self.n_envs} > {self.target_update_interval}), "
                    "therefore the target network will be updated after each call to env.step() "
                    f"which corresponds to {self.n_envs} steps."
                )

        if self.prioritized_er:
            # gamma for n-step bootstrapping
            self.gamma = self.gamma**self.replay_buffer.n_step

    def _create_aliases(self) -> None:
        self.q_net = self.policy.q_net
        self.q_net_target = self.policy.q_net_target

    def _on_step(self) -> None:
        """
        Update the exploration rate, PER beta and target network if needed.

        This method is called in ``collect_rollouts()`` after each step in the environment.

        Pretty much a copy of stable baseline's DQN._on_step, but making use of the
        num_timesteps counter instead of a custom _n_calls counter, and also updating
        PER buffer beta if needed
        """

        # Account for multiple environments
        # each call to step() corresponds to n_envs transitions
        if self.num_timesteps % self.target_update_interval == 0:
            polyak_update(
                self.q_net.parameters(), self.q_net_target.parameters(), self.tau
            )
            # Copy running stats, see GH issue #996
            polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self.exploration_rate = self.exploration_schedule(
            self._current_progress_remaining
        )
        self.logger.record("rollout/exploration_rate", self.exploration_rate)

        if self.prioritized_er:
            self.replay_buffer.update_beta(self._current_progress_remaining)
            self.logger.record("rollout/PER_beta", self.replay_buffer.beta)

    @torch.no_grad()
    def _td_target(self, reward: Tensor, next_state: Tensor, done: Tensor) -> Tensor:
        """Compute the q-learning temporal difference target

        If double q-learning is enabled (double_dqn flag) then the online network
        chooses the actions for the next state, which are then evaluated using the target network.
        Otherwise, the max q-value of the online network is used.

        Finally, the current reward and the estimated next q-value are combined to form the TD target

        Args:
            reward (Tensor): reward gained by going from current state to next state (float)
            next_state (Tensor): next state / observations
            done (Tensor): done flags to annotate the episode as having ended. If True/1, the next q-value is not used

        Returns:
            Tensor: Temporal difference targets
        """
        # resample the noisy linear weights
        if self.noisy_dqn:
            reset_noise(self.q_net_target)

        if self.double_dqn:
            best_action = torch.argmax(
                self.q_net(next_state, advantages_only=True), dim=1
            )  # could maybe replace with self.policy._predict
            next_Q = torch.gather(
                self.q_net_target(next_state), dim=1, index=best_action.unsqueeze(1)
            ).squeeze()
        else:
            next_Q = torch.max(self.q_net_target(next_state), dim=1)[0]

        return reward + self.gamma * next_Q * (1 - done)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        grad_norms = []
        qs = []
        for train_iter in range(gradient_steps):
            if self.noisy_dqn and train_iter > 0:
                reset_noise(self.q_net)

            # Sample replay buffer
            replay_data = self.replay_buffer.sample(
                batch_size, env=self._vec_normalize_env
            )
            if self.prioritized_er:
                # if PER is being used, the return type is actually a tuple, so split it correctly
                indices, weights, replay_data = replay_data
                weights = torch.from_numpy(weights).cuda()

            with autocast(enabled=self.use_amp):
                td_est = torch.gather(
                    self.q_net(replay_data.observations),
                    dim=1,
                    index=replay_data.actions,
                ).squeeze()
                td_tgt = self._td_target(
                    replay_data.rewards.squeeze(1),
                    replay_data.next_observations,
                    replay_data.dones.squeeze(1),
                )

                self.policy.optimizer.zero_grad()
                if self.prioritized_er:
                    # update PER priorities
                    td_errors = td_est - td_tgt
                    # 1e-6 is the epsilon in PER
                    new_priorities = np.abs(td_errors.detach().cpu().numpy()) + 1e-6
                    self.replay_buffer.update_priorities(indices, new_priorities)

                    # compute weighted loss
                    cur_losses = self.loss_fn(td_tgt, td_est)
                    loss = torch.mean(weights * cur_losses)
                else:
                    # compute normal mean loss
                    loss = self.loss_fn(td_tgt, td_est)

            # Optimize the policy

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.policy.optimizer)
            # Clip gradient norm
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.max_grad_norm
            )
            # update online network and scaler
            self.scaler.step(self.policy.optimizer)
            self.scaler.update()

            losses.append(loss.item())
            grad_norms.append(grad_norm.item())
            qs.append(td_est.mean().item())

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))
        self.logger.record("train/grad_norm", np.mean(grad_norms))
        self.logger.record("train/q_value", np.mean(qs))
        self.logger.dump(self.num_timesteps)

    @torch.no_grad
    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Tuple[np.ndarray, ...] | None = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Tuple[np.ndarray, ...] | None]:
        """Overrides the base_class predict function to include epsilon-greedy exploration.

        This is a copy of the stable baseline DQN.predict function.

        Args:
            observation (Union[np.ndarray, Dict[str, np.ndarray]]): the input observation
            state (Tuple[np.ndarray, ...] | None, optional): The last states (used in recurrent policies). Defaults to None.
            episode_start (np.ndarray | None, optional): The last masks (used in recurrent policies). Defaults to None.
            deterministic (bool, optional): Whether or not to return deterministic actions. Defaults to False.

        Returns:
            (action, next_hidden) (Tuple[np.ndarray, Tuple[np.ndarray, ...] | None]): the model's action
                and the next hidden state (used in recurrent policies)
        """
        if not deterministic and np.random.rand() < self.exploration_rate:
            if is_vectorized_observation(
                maybe_transpose(observation, self.observation_space),
                self.observation_space,
            ):
                if isinstance(self.observation_space, spaces.Dict):
                    n_batch = observation[list(observation.keys())[0]].shape[0]
                else:
                    n_batch = observation.shape[0]
                action = np.array([self.action_space.sample() for _ in range(n_batch)])
            else:
                action = np.array(self.action_space.sample())
        else:
            with autocast(enabled=self.use_amp):
                action, state = self.policy.predict(
                    observation, state, episode_start, deterministic
                )
        return action, state

    def learn(
        self: SelfRainbow,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "Rainbow",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfRainbow:
        # if noisy linears are used, add callback for resampling the noise
        if self.noisy_dqn:
            if callback is None:
                callback = NoiseReset()
            else:
                callback = [NoiseReset()] + (
                    callback if isinstance(callback, list) else [callback]
                )

        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _excluded_save_params(self) -> List[str]:
        return super()._excluded_save_params() + [
            "q_net",
            "q_net_target",
            "policy.optimizer.step_calcs",
            "policy.optimizer.sample_weights",
        ]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []

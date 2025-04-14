# TODO add/fix docstrings
# TODO add comments

import random
import warnings
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as torch
from torch.cuda.amp import GradScaler, autocast
from gym import spaces

from stable_baselines3.common.buffers import BaseBuffer
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.callbacks import BaseCallback

from async_algorithm import AsyncOffPolicyAlgorithm
from policies import CnnPolicy, RainbowPolicy, MlpPolicy, RainbowNetwork
from torch_layers import FactorizedNoisyLinear
from buffer import PrioritizedReplayBuffer

SelfRainbow = TypeVar("SelfRainbow", bound="Rainbow")


@torch.no_grad()
def reset_noise(net) -> None:
    for m in net.modules():
        if isinstance(m, FactorizedNoisyLinear):
            m.reset_noise()


class NoiseReset(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

    def _on_rollout_start(self) -> None:
        reset_noise(self.model.q_net)

    def _on_step(self) -> bool:
        return True


class Rainbow(AsyncOffPolicyAlgorithm):
    """
    Deep Q-Network (DQN)

    Paper: https://arxiv.org/abs/1312.5602, https://www.nature.com/articles/nature14236
    Default hyperparameters are taken from the Nature paper,
    except for the optimizer and learning rate that were taken from Stable Baselines defaults.

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1) default 1 for hard update
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param target_update_interval: update the target network every ``target_update_interval``
        environment steps.
    :param exploration_fraction: fraction of entire training period over which the exploration rate is reduced
    :param exploration_initial_eps: initial value of random action probability
    :param exploration_final_eps: final value of random action probability
    :param max_grad_norm: The maximum value for the gradient clipping
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
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
        env: Union[GymEnv, str],  # parallel_envs, subproc_vecenv
        learning_rate: Union[
            float, Schedule
        ] = 0.00025,  # lr, lr_decay_steps, lr_decay_factor
        buffer_size: int = int(2**20),  # capacity
        learning_starts: int = 100_000,  # burnin
        batch_size: int = 256,  # batch_size
        tau: float = 1.0,
        gamma: float = 0.99,  # gamma
        train_freq: Union[int, Tuple[int, str]] = (1, "step"),  # train after every step
        gradient_steps: int = 2,  # train_count
        replay_buffer_class: Optional[
            Type[BaseBuffer]
        ] = PrioritizedReplayBuffer,  # prioritized_er
        replay_buffer_kwargs: Optional[Dict[str, Any]] = {"n_step": 3},  # n_step
        optimize_memory_usage: bool = False,  # ignored
        prioritized_er_beta0_initial: float = 0.45,  # 0.4 for rainbow, 0.5 for dopamine
        prioritized_er_beta0_final: float = 1.0,  # set prioritized_er_beta0_initial == prioritized_er_beta0_final for constant value
        prioritized_er_beta0_fraction: float = 1.0,
        target_update_interval: int = 32_000,  # sync_dqn_target_every
        exploration_fraction: float = 0.05,  # eps_decay_frames
        exploration_initial_eps: float = 1.0,  # init_eps
        exploration_final_eps: float = 0.01,  # final_eps
        double_dqn: bool = True,
        max_grad_norm: float = 10,  # max_grad_norm
        loss_fn: str = "huber",  # loss_fn
        # stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = {
            "noisy_linear": True,
            "linear_kwargs": {"sigma_0": 0.5},
            "optimizer_kwargs": {"eps": None},
        },  # noisy_dqn, noisy_sigma0, adam_eps
        verbose: int = 0,
        seed: Optional[int] = None,
        use_amp: bool = True,
        device: Union[torch.device, str] = "auto",
        _init_setup_model: bool = True,
    ) -> None:
        if replay_buffer_kwargs is not None:
            replay_buffer_kwargs["gamma"] = gamma
            replay_buffer_kwargs["use_amp"] = use_amp

        if policy_kwargs["optimizer_kwargs"]["eps"] is None:
            policy_kwargs["optimizer_kwargs"]["eps"] = 0.005 / batch_size
            print("Optimizer eps:", policy_kwargs["optimizer_kwargs"]["eps"])

        if policy_kwargs["noisy_linear"]:
            exploration_initial_eps = 0.002
            exploration_final_eps = 0.0
            exploration_fraction = 0.002  # 20_000 time steps with 10M training steps. may need to be adjusted if training length changes

        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
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

        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.exploration_fraction = exploration_fraction

        self.per_beta_initial = prioritized_er_beta0_initial
        self.per_beta_final = prioritized_er_beta0_final
        self.per_beta_fraction = prioritized_er_beta0_fraction  # controls when the final value is reached. e.g. 0.1 means 10% before the training finishes

        self.exploration_rate = exploration_initial_eps
        self.per_beta = prioritized_er_beta0_initial

        self.target_update_interval = target_update_interval
        # For updating the target network with multiple envs:
        assert self.target_update_interval % self.n_envs == 0, (
            "target would not be synced since the main loop iterates in steps of parallel_envs"
        )
        self._game_frame = 0
        self.max_grad_norm = max_grad_norm

        self.double_dqn = double_dqn
        # n_step for n step bootstrapping (buffers)
        self.n_step_gamma = self.gamma ** replay_buffer_kwargs["n_step"]
        self.noisy_dqn = policy_kwargs["noisy_linear"]

        self.prioritized_er = replay_buffer_class == PrioritizedReplayBuffer
        loss_fn_cls = torch.nn.MSELoss if loss_fn == "mse" else torch.nn.SmoothL1Loss
        self.loss_fn = loss_fn_cls(
            reduction=("none" if self.prioritized_er else "mean")
        )

        self.use_amp = use_amp
        self.scaler = GradScaler(enabled=self.use_amp)

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()

        self.exploration_schedule = get_linear_fn(
            self.exploration_initial_eps,
            self.exploration_final_eps,
            self.exploration_fraction,
        )
        self.per_beta_schedule = get_linear_fn(
            self.per_beta_initial,
            self.per_beta_final,
            self.per_beta_fraction,
        )

        if self.n_envs > 1:
            if self.n_envs > self.target_update_interval:
                warnings.warn(
                    "The number of environments used is greater than the target network "
                    f"update interval ({self.n_envs} > {self.target_update_interval}), "
                    "therefore the target network will be updated after each call to env.step() "
                    f"which corresponds to {self.n_envs} steps."
                )

    def _create_aliases(self) -> None:
        self.q_net = self.policy.q_net
        self.q_net_target = self.policy.q_net_target

    def _on_step(self) -> None:
        """
        Update the exploration rate and target network if needed.
        This method is called in ``collect_rollouts()`` after each step in the environment.
        """
        # Account for multiple environments
        # each call to step() corresponds to n_envs transitions

        if (
            (self.num_timesteps - self.n_envs) % self.target_update_interval == 0
            and self.replay_buffer.size() > self.learning_starts
        ):
            self.q_net_target.load_state_dict(self.q_net.state_dict())

        self.exploration_rate = self.exploration_schedule(
            self._current_progress_remaining
        )
        self.per_beta = self.per_beta_schedule(self._current_progress_remaining)
        self.logger.record("rollout/exploration_rate", self.exploration_rate)
        self.logger.record("rollout/prioritized_er_beta", self.per_beta)

    @torch.no_grad()
    def _td_target(self, reward: float, next_state, done: bool):
        reset_noise(self.q_net_target)
        if self.double_dqn:
            best_action = torch.argmax(
                self.q_net(next_state, advantages_only=True), dim=1
            )  # could maybe replace with self.policy._predict
            next_Q = torch.gather(
                self.q_net_target(next_state), dim=1, index=best_action.unsqueeze(1)
            ).squeeze()
            return reward + self.n_step_gamma * next_Q * (1 - done)
        else:
            max_q = torch.max(self.q_net_target(next_state), dim=1)[0]
            return reward + self.n_step_gamma * max_q * (1 - done)

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
            # replay_data (observations, next_observations, actions, rewards, dones) : (state, next_state, action, reward, done)
            if self.prioritized_er:
                indices, weights, replay_data = self.replay_buffer.sample(
                    batch_size, self.per_beta, env=self._vec_normalize_env
                )
                weights = torch.from_numpy(weights).cuda()
            else:
                replay_data = self.replay_buffer.sample(
                    batch_size, env=self._vec_normalize_env
                )
            # replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            with autocast(enabled=self.use_amp):
                # removed replay_data.actions.unsqueeze(1)
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
                    td_errors = td_est - td_tgt
                    new_priorities = (
                        np.abs(td_errors.detach().cpu().numpy()) + 1e-6
                    )  # 1e-6 is the epsilon in PER
                    self.replay_buffer.update_priorities(indices, new_priorities)

                    cur_losses = self.loss_fn(td_tgt, td_est)
                    loss = torch.mean(weights * cur_losses)
                else:
                    loss = self.loss_fn(td_tgt, td_est)

            # Optimize the policy

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.policy.optimizer)
            # Clip gradient norm
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.max_grad_norm
            )
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

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Overrides the base_class predict function to include epsilon-greedy exploration.

        :param observation: the input observation
        :param state: The last states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next state
            (used in recurrent policies)
        """

        with torch.no_grad():
            with autocast(enabled=self.use_amp):
                action, state = self.policy.predict(
                    observation, state, episode_start, deterministic
                )
                if not deterministic and self.exploration_rate > 0:
                    for i in range(action.shape[0]):
                        if random.random() < self.exploration_rate:
                            action[i] = self.action_space.sample()

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
        return super().learn(
            total_timesteps=total_timesteps,
            callback=NoiseReset()
            if callback is None
            else [NoiseReset()]
            + (callback if isinstance(callback, list) else [callback]),
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _excluded_save_params(self) -> List[str]:
        return [
            *super()._excluded_save_params(),
            "q_net",
            "q_net_target",
            "policy.optimizer.step_calcs",
            "policy.optimizer.sample_weights",
        ]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []

import random
from collections import deque
from math import sqrt
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
from gym import spaces

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import (
    ReplayBufferSamples,
)
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.utils import get_linear_fn


# making this a subclass of ReplayBuffer is simply to give an idea of what it does
# it actually does not use any of the code in ReplayBuffer and could just be a subclass of BaseBuffer
class PrioritizedReplayBuffer(ReplayBuffer):
    """originally based on https://nn.labml.ai/rl/dqn, supports n-step bootstrapping and parallel environments,
    removed alpha hyperparameter like google/dopamine (set to 0.5, i.e. sqrt)

    this version is primarily based on https://github.com/schmidtdominik/Rainbow/blob/298c93d3d9322440d3a22cf24045b57af9c83fde/common/replay_buffer.py

    adapted to approximately follow the stable baselines 3 Buffer design. Of course this was not completely possible,
    as for example the training algorithm also needs to be made aware of the weights,
    for which there is no inherent support in sb3
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[torch.device, str] = "auto",
        n_envs: int = 1,
        n_step: int = 3,
        gamma: float = 0.99,
        beta_initial: float = 0.45,
        beta_final: float = 1.0,
        beta_end_fraction: float = 1.0,
        optimize_memory_usage=False,
    ):
        """
        Args:
            buffer_size (int): Max number of elements in the buffer
            observation_space (spaces.Space): Observation space of the environment
            action_space (spaces.Space): Action space of the environment
            device (Union[torch.device, str], optional): PyTorch device. Defaults to "auto".
            n_envs (int, optional): Number of parallel environments. Defaults to 1.
            n_step (int, optional): perform n-step bootstrapping by returning transitions that
                cover ``n_step``s in the environment. Defaults to 3.
            gamma (float, optional): the discount factor for a single step,
                used to calculate n-step transitions. Defaults to 0.99.
            beta_initial (float, optional): Initial beta value for weighted importance sampling.
                0.4 for rainbow, 0.5 for dopamine. Defaults to 0.45.
            beta_final (float, optional): Final beta value, should always be 1 to prevent bias. Defaults to 1.0.
            beta_end_fraction (float, optional): fraction of entire training period over
                which beta is reduced. Defaults to 1.0.
            optimize_memory_usage (bool, optional): Ignored. Simply here to fit the automatic sb3 instantiation
        """
        # use super(ReplayBuffer, self) instead of super() to skip the ReplayBuffer.__init__
        # that includes a lot of stuff that is not needed here
        super(ReplayBuffer, self).__init__(
            buffer_size, observation_space, action_space, device, n_envs=n_envs
        )

        self.beta_schedule = get_linear_fn(beta_initial, beta_final, beta_end_fraction)
        self.beta = beta_initial

        self.n_step = n_step
        self.gamma = gamma

        # define buffers of type object, which can be used to index and slice data
        # without evaluating the objects as arrays, thus preserving lazy frames
        self.observations = np.full(self.buffer_size, None)
        self.next_observations = np.full(self.buffer_size, None)

        self.actions = torch.zeros(
            (self.buffer_size, self.action_dim), dtype=torch.int64, device=self.device
        )

        self.rewards = torch.zeros(
            (self.buffer_size,), dtype=torch.float32, device=self.device
        )
        self.dones = torch.zeros(
            (self.buffer_size,), dtype=torch.float32, device=self.device
        )

        # queues used to construct n-step tranitions
        self.n_step_buffers = [
            deque(maxlen=self.n_step + 1) for j in range(self.n_envs)
        ]

        self.priority_sum = np.full((2 * self.buffer_size), 0.0, dtype=np.float64)
        self.priority_min = np.full((2 * self.buffer_size), np.inf, dtype=np.float64)

        self.max_priority = 1.0  # initial priority of new transitions

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
            next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        for queue, o, a, r, d in zip(self.n_step_buffers, obs, action, reward, done):
            queue.append((o, a, r, d))

            # n-step transition can't start on a terminal state
            if len(queue) == self.n_step + 1 and not queue[0][3]:
                # get first and last states of the n_step stransition
                o, a, r, _ = queue[0]
                no, _, _, d = queue[self.n_step]

                # calculate reward between
                for k in range(1, self.n_step):
                    r += queue[k][2] * self.gamma**k
                    if queue[k][3]:
                        d = True
                        break

                # store observation data without evaluating it
                self.observations[self.pos] = o
                self.next_observations[self.pos] = no

                # store scalar data
                self.actions[self.pos] = self.to_torch(a)
                self.rewards[self.pos] = self.to_torch(r)
                self.dones[self.pos] = self.to_torch(d)

                # set initial priority
                self._set_priority_min(self.pos, sqrt(self.max_priority))
                self._set_priority_sum(self.pos, sqrt(self.max_priority))

                self.pos += 1
                if self.pos == self.buffer_size:
                    self.full = True
                    self.pos = 0

    def _set_priority_min(self, idx, priority_alpha):
        idx += self.buffer_size
        self.priority_min[idx] = priority_alpha
        while idx >= 2:
            idx //= 2
            self.priority_min[idx] = min(
                self.priority_min[2 * idx], self.priority_min[2 * idx + 1]
            )

    def _set_priority_sum(self, idx, priority):
        idx += self.buffer_size
        self.priority_sum[idx] = priority
        while idx >= 2:
            idx //= 2
            self.priority_sum[idx] = (
                self.priority_sum[2 * idx] + self.priority_sum[2 * idx + 1]
            )

    def _sum(self):
        return self.priority_sum[1]

    def _min(self):
        return self.priority_min[1]

    def find_prefix_sum_idx(self, prefix_sum):
        """find the largest i such that the sum of the leaves from 1 to i is <= prefix sum"""

        idx = 1
        while idx < self.buffer_size:
            if self.priority_sum[idx * 2] > prefix_sum:
                idx = 2 * idx
            else:
                prefix_sum -= self.priority_sum[idx * 2]
                idx = 2 * idx + 1
        return idx - self.buffer_size

    def update_priorities(self, indexes, priorities):
        for idx, priority in zip(indexes, priorities):
            self.max_priority = max(self.max_priority, priority)
            priority_alpha = sqrt(priority)
            self._set_priority_min(idx, priority_alpha)
            self._set_priority_sum(idx, priority_alpha)

    def update_beta(self, current_progress_remaining: float):
        self.beta = self.beta_schedule(current_progress_remaining)

    def sample(
        self, batch_size: int, env: Union[VecNormalize, None] = None
    ) -> Tuple[np.ndarray, torch.Tensor, ReplayBufferSamples]:
        """Sample from the prioritized transition distribution

        Args:
            batch_size (int): Number of elements to sample
            env (Union[VecNormalize, None], optional): Ignored. Not supported at the moment

        Returns:
            (indices, weights, samples) (Tuple[np.ndarray, torch.Tensor, ReplayBufferSamples]):
                The usual replay buffer samples along with their indices in the buffer and corresponding weights
        """
        weights = np.zeros(shape=batch_size, dtype=np.float32)
        indices = np.zeros(shape=batch_size, dtype=np.int32)

        for i in range(batch_size):
            p = random.random() * self._sum()
            idx = self.find_prefix_sum_idx(p)
            indices[i] = idx

        prob_min = self._min() / self._sum()
        max_weight = (prob_min * self.size()) ** (-self.beta)

        for i in range(batch_size):
            idx = indices[i]
            prob = self.priority_sum[idx + self.buffer_size] / self._sum()
            weight = (prob * self.size()) ** (-self.beta)
            weights[i] = weight / max_weight

        return (
            indices,
            torch.from_numpy(weights).to(self.device),
            self._get_samples(indices, env),
        )

    def obs_to_torch(self, array, copy=True):
        array = np.stack([np.array(obs, copy=False) for obs in array])
        return super().to_torch(array, copy)

    def _get_samples(
        self, batch_inds: np.ndarray, env: VecNormalize | None = None
    ) -> ReplayBufferSamples:
        return ReplayBufferSamples(
            self.obs_to_torch(self.observations[batch_inds]),
            self.actions[batch_inds, :],
            self.obs_to_torch(self.next_observations[batch_inds]),
            self.dones[batch_inds].reshape(-1, 1),
            self.rewards[batch_inds].reshape(-1, 1),
        )

    def reset(self):
        # reset pos and full. Causes e.g. observations to be automatically overwritten, so need to reset them
        super().reset()

        self.n_step_buffers = [
            deque(maxlen=self.n_step + 1) for j in range(self.n_envs)
        ]

        self.priority_sum = np.full((2 * self.buffer_size), 0.0, dtype=np.float64)
        self.priority_min = np.full((2 * self.buffer_size), np.inf, dtype=np.float64)

        self.max_priority = 1.0

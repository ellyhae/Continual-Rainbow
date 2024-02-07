# TODO: implement reset functionality. https://github.com/DLR-RM/stable-baselines3/blob/9cebedc89fcabf509241d89b09bf39fb61cab94b/stable_baselines3/common/buffers.py#L95

import warnings
from typing import Any, Dict, Generator, List, Optional, Union

from collections import deque
import random
from math import sqrt

import numpy as np
import torch
from gym import spaces

from stable_baselines3.common.buffers import BaseBuffer
from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from stable_baselines3.common.type_aliases import (
    DictReplayBufferSamples,
    DictRolloutBufferSamples,
    ReplayBufferSamples,
    RolloutBufferSamples,
)
from stable_baselines3.common.vec_env import VecNormalize

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None
    
from utils import prep_observation_for_qnet

#class UniformReplayBuffer(BaseBuffer):
#    """
#    Replay buffer used in off-policy algorithms like SAC/TD3.
#    :param buffer_size: Max number of element in the buffer
#    :param observation_space: Observation space
#    :param action_space: Action space
#    :param device: PyTorch device
#    :param n_envs: Number of parallel environments
#    :param optimize_memory_usage: Enable a memory efficient variant
#        of the replay buffer which reduces by almost a factor two the memory used,
#        at a cost of more complexity.
#        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
#        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
#        Cannot be used in combination with handle_timeout_termination.
#    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
#        separately and treat the task as infinite horizon task.
#        https://github.com/DLR-RM/stable-baselines3/issues/284
#    """
#
#    def __init__(
#        self,
#        buffer_size: int,
#        observation_space: spaces.Space,
#        action_space: spaces.Space,
#        device: Union[torch.device, str] = "auto",
#        n_envs: int = 1,
#        n_step: int = 3,
#        gamma: float = 0.99,
#        optimize_memory_usage=False, # ignored
#    ):
#        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)
#
#        self.n_step = n_step
#        self.gamma = gamma
#
#        # Check that the replay buffer can fit into the memory
#        if psutil is not None:
#            mem_available = psutil.virtual_memory().available
#
#        self.observations = np.zeros((self.buffer_size, *self.obs_shape), dtype=observation_space.dtype)
#        self.next_observations = np.zeros((self.buffer_size, *self.obs_shape), dtype=observation_space.dtype)
#
#        self.actions = np.zeros((self.buffer_size, self.action_dim), dtype=action_space.dtype)
#
#        self.rewards = np.zeros((self.buffer_size,), dtype=np.float32)
#        self.dones = np.zeros((self.buffer_size,), dtype=np.float32)
#
#        self.n_step_buffers = [deque(maxlen=self.n_step + 1) for j in range(self.n_envs)]
#
#        if psutil is not None:
#            total_memory_usage = self.observations.nbytes + self.next_observations.nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes
#
#            if total_memory_usage > mem_available:
#                # Convert to GB
#                total_memory_usage /= 1e9
#                mem_available /= 1e9
#                warnings.warn(
#                    "This system does not have apparently enough memory to store the complete "
#                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
#                )
#
#    def add(
#        self,
#        obs: np.ndarray,
#        next_obs: np.ndarray,
#        action: np.ndarray,
#        reward: np.ndarray,
#        done: np.ndarray,
#        infos: List[Dict[str, Any]],
#    ) -> None:
#        # Reshape needed when using multiple envs with discrete observations
#        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
#        if isinstance(self.observation_space, spaces.Discrete):
#            obs = obs.reshape((self.n_envs, *self.obs_shape))
#            next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))
#
#        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
#        action = action.reshape((self.n_envs, self.action_dim))
#
#        for queue, o, a, r, d in zip(self.n_step_buffers, obs, action, reward, done):
#            queue.append((o, a, r, d))
#            if len(queue) == self.n_step + 1 and not d:
#                # get first and last states of the n_step stransition
#                o, a, r, _ = queue[0]
#                no, _, _, d = queue[-1]
#
#                # calculate reward between
#                for k in range(1, self.n_step):
#                    r += queue[k][2] * self.gamma ** k
#                    if queue[k][3]:
#                        d = True
#                        break
#
#                self.observations[self.pos] = np.array(o).copy()
#                self.next_observations[self.pos] = np.array(no).copy()
#
#                self.actions[self.pos] = np.array(a).copy()
#                self.rewards[self.pos] = np.array(r).copy()
#                self.dones[self.pos] = np.array(d).copy()
#
#                self.pos += 1
#                if self.pos == self.buffer_size:
#                    self.full = True
#                    self.pos = 0
#
#    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
#
#        data = (
#            self._normalize_obs(self.observations[batch_inds, :], env),
#            self.actions[batch_inds, :],
#            self._normalize_obs(self.next_observations[batch_inds, :], env),
#            self.dones[batch_inds].reshape(-1, 1),
#            self._normalize_reward(self.rewards[batch_inds].reshape(-1, 1), env),
#        )
#        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))
    

class PrioritizedReplayBuffer(BaseBuffer):
    """
    Replay buffer used in off-policy algorithms like SAC/TD3.
    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
        Cannot be used in combination with handle_timeout_termination.
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
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
        use_amp: bool = True,
        optimize_memory_usage=False, # ignored
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)

        self.n_step = n_step
        self.gamma = gamma
        self.use_amp = use_amp

        # Check that the replay buffer can fit into the memory
        #if psutil is not None:
        #    mem_available = psutil.virtual_memory().available

        #self.observations = np.zeros((self.buffer_size, *self.obs_shape), dtype=observation_space.dtype)
        #self.next_observations = np.zeros((self.buffer_size, *self.obs_shape), dtype=observation_space.dtype)
        
        self.observations = np.full(self.buffer_size, None)
        self.next_observations = np.full(self.buffer_size, None)

        #self.actions = np.zeros((self.buffer_size, self.action_dim), dtype=action_space.dtype)

        #self.rewards = np.zeros((self.buffer_size,), dtype=np.float32)
        #self.dones = np.zeros((self.buffer_size,), dtype=np.float32)
        
        self.actions = torch.zeros((self.buffer_size, self.action_dim), dtype=torch.int64, device=self.device)
        
        self.rewards = torch.zeros((self.buffer_size,), dtype=torch.float32, device=self.device)
        self.dones = torch.zeros((self.buffer_size,), dtype=torch.float32, device=self.device)

        self.n_step_buffers = [deque(maxlen=self.n_step + 1) for j in range(self.n_envs)]

        self.priority_sum = np.full((2 * self.buffer_size), 0., dtype=np.float64)
        self.priority_min = np.full((2 * self.buffer_size), np.inf, dtype=np.float64)
        
        #self.priority_sum = [0 for _ in range(2 * self.buffer_size)]
        #self.priority_min = [float('inf') for _ in range(2 * self.buffer_size)]

        self.max_priority = 1.0  # initial priority of new transitions

        # not very useful, as observations are stored in a memory efficient way
        #if psutil is not None:
        #    total_memory_usage = self.observations.nbytes + self.next_observations.nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes
        #    if total_memory_usage > mem_available:
        #        # Convert to GB
        #        total_memory_usage /= 1e9
        #        mem_available /= 1e9
        #        warnings.warn(
        #            "This system does not have apparently enough memory to store the complete "
        #            f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
        #        )

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
            #next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        for queue, o, a, r, d in zip(self.n_step_buffers, obs, action, reward, done):
            queue.append((o, a, r, d))
            if len(queue) == self.n_step + 1 and not queue[0][3]:  # n-step transition can't start on a terminal state
                # get first and last states of the n_step stransition
                o, a, r, _ = queue[0]
                no, _, _, d = queue[self.n_step]

                # calculate reward between
                for k in range(1, self.n_step):
                    r += queue[k][2] * self.gamma ** k
                    if queue[k][3]:
                        d = True
                        break

                self.observations[self.pos] = o #np.array(o).copy()
                self.next_observations[self.pos] = no #np.array(no).copy()

                self.actions[self.pos] = self.to_torch(a) # np.array(a).copy()
                self.rewards[self.pos] = self.to_torch(r) # np.array(r).copy()
                self.dones[self.pos] = self.to_torch(d) # np.array(d).copy()
                
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
            self.priority_min[idx] = min(self.priority_min[2 * idx], self.priority_min[2 * idx + 1])

    def _set_priority_sum(self, idx, priority):
        idx += self.buffer_size
        self.priority_sum[idx] = priority
        while idx >= 2:
            idx //= 2
            self.priority_sum[idx] = self.priority_sum[2 * idx] + self.priority_sum[2 * idx + 1]

    def _sum(self):
        return self.priority_sum[1]

    def _min(self):
        return self.priority_min[1]
    
    def find_prefix_sum_idx(self, prefix_sum):
        """ find the largest i such that the sum of the leaves from 1 to i is <= prefix sum"""

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

    def sample(self, batch_size: int, beta: float, env: Union[VecNormalize, None] = None):
        weights = np.zeros(shape=batch_size, dtype=np.float32)
        indices = np.zeros(shape=batch_size, dtype=np.int32)

        for i in range(batch_size):
            p = random.random() * self._sum()
            idx = self.find_prefix_sum_idx(p)
            indices[i] = idx

        prob_min = self._min() / self._sum()
        max_weight = (prob_min * self.size()) ** (-beta)

        for i in range(batch_size):
            idx = indices[i]
            prob = self.priority_sum[idx + self.buffer_size] / self._sum()
            weight = (prob * self.size()) ** (-beta)
            weights[i] = weight / max_weight

        return indices, weights, self._get_samples(indices, env)

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:

        stack_lazy = lambda data: torch.stack(list(map(lambda x: torch.from_numpy(x.__array__()), data)))
        
        data = (
            prep_observation_for_qnet(stack_lazy(self.observations[batch_inds]), self.use_amp),
            self.actions[batch_inds, :],
            prep_observation_for_qnet(stack_lazy(self.next_observations[batch_inds]), self.use_amp),
            self.dones[batch_inds].reshape(-1, 1),
            self.rewards[batch_inds].reshape(-1, 1),
        )
        return ReplayBufferSamples(*data)
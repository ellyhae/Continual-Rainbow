"""
This is a slightly modified copy of https://github.com/schmidtdominik/Rainbow/blob/main/common/vec_envs.py

Note from the original file:
This file handles some of the internals for vectorized environments.
"""

from collections import deque
from copy import deepcopy

from gym.spaces import Box
from gym.wrappers import LazyFrames

from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnvWrapper

import numpy as np


class DummyVecEnvNoFlatten(DummyVecEnv):
    """
    Slightly modified version of stable_baselines3's DummyVecEnv. The main difference is that observations are not
    flattened before they are returned. This is done to make it work with the lazy frame-stacking class further below.
    """

    def step_wait(self):
        obs_list = []
        for env_idx in range(self.num_envs):
            (
                obs,
                self.buf_rews[env_idx],
                self.buf_dones[env_idx],
                self.buf_infos[env_idx],
            ) = self.envs[env_idx].step(self.actions[env_idx])
            if self.buf_dones[env_idx]:
                # save final observation where user can get it, then reset
                self.buf_infos[env_idx]["terminal_observation"] = obs
                obs = self.envs[env_idx].reset()
            obs_list.append(obs)
        return (
            obs_list,
            np.copy(self.buf_rews),
            np.copy(self.buf_dones),
            deepcopy(self.buf_infos),
        )

    def reset(self):
        obs_list = []
        for env_idx in range(self.num_envs):
            obs = self.envs[env_idx].reset()
            obs_list.append(obs)
        return obs_list


class SubprocVecEnvNoFlatten(SubprocVecEnv):
    """
    Slightly modified version of stable_baselines3's SubprocVecEnv. The main difference is that observations are not
    flattened before they are returned. This is done to make it work with the lazy frame-stacking class further below.
    """

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return obs, np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(("reset", None))
        obs = [remote.recv() for remote in self.remotes]
        return obs


class LazyVecFrameStack(VecEnvWrapper):
    """
    Lazy & vectorized frame stacking implementation based on OpenAI-Baselines FrameStack and Stable-Baselines-3 VecFrameStack wrappers.

    Documentation of FrameStack:
    Observation wrapper that stacks the observations in a rolling manner.

    For example, if the number of stacks is 4, then the returned observation contains
    the most recent 4 observations. For environment 'Pendulum-v1', the original observation
    is an array with shape [3], so if we stack 4 observations, the processed observation
    has shape [4, 3].

    Args:
        venv (VecEnv): environment object
        num_stack (int): number of stacks
        lz4_compress (bool): use lz4 to compress the frames internally
    """

    def __init__(
        self,
        venv: VecEnv,
        num_stack: int,
        clone_arrays: bool,
        lz4_compress: bool = False,
    ):
        super().__init__(venv)
        self.num_stack = num_stack
        self.lz4_compress = lz4_compress
        self.clone_arrays = clone_arrays

        self.frames = [deque(maxlen=num_stack) for _ in range(self.num_envs)]

        low = np.repeat(self.observation_space.low[np.newaxis, ...], num_stack, axis=0)
        high = np.repeat(
            self.observation_space.high[np.newaxis, ...], num_stack, axis=0
        )
        self.observation_space = Box(
            low=low, high=high, dtype=self.observation_space.dtype
        )

    def _get_observation(self):
        result = []
        for i in range(len(self.frames)):
            assert len(self.frames[i]) == self.num_stack, (
                len(self.frames[i]),
                self.num_stack,
            )
            result.append(LazyFrames(list(self.frames[i]), self.lz4_compress))
        return result

    def step_wait(self):
        observations, rewards, dones, infos = self.venv.step_wait()

        # Note: copying all the arrays here is necessary to prevent some weird memory leak when using procgen with DummyVecEnv
        # (SubprocVecenv copies the arrays anyway when moving them to the main process)
        if self.clone_arrays:
            for i, observation in enumerate(observations):
                self.frames[i].append(observation.copy())
            return self._get_observation(), rewards.copy(), dones.copy(), infos.copy()
        else:
            for i, observation in enumerate(observations):
                self.frames[i].append(observation)
            return self._get_observation(), rewards, dones, infos

    def reset(self, **kwargs):
        observations = self.venv.reset(**kwargs)

        for i, observation in enumerate(observations):
            for _ in range(self.num_stack):
                self.frames[i].append(observation)
        return self._get_observation()

    def close(self) -> None:
        self.venv.close()

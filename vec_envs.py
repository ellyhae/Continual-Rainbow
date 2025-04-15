"""
This is a slightly modified copy of https://github.com/schmidtdominik/Rainbow/blob/main/common/vec_envs.py

Note from the original file:
This file handles some of the internals for vectorized environments.
"""

from typing import List, Tuple, Dict

from collections import deque
from copy import deepcopy

from gym.spaces import Box

from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnvWrapper

import numpy as np

try:
    from lz4.block import compress, decompress
except ImportError:
    compress = decompress = None


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


class LazyStackedObservations:
    """A slightly modified version of gym's LazyFrames,
    which stacks (concatenates) along an existing axis instad of creating a new axis.

    Ensures common frames are only stored once to optimize memory use."""

    def __init__(
        self, frames: List[np.ndarray], lz4_compress: bool = False, stack_axis: int = -1
    ):
        """
        Args:
            frames (List[np.ndarray]): observation data to be stored
            lz4_compress (bool, optional): use lz4 to compress the frames internally. Defaults to False.
            stack_axis (int, optional): channel axis for the observation data, along which successive frames are stacked.
                                        Defaults to -1.
        """
        self.frame_shape = tuple(frames[0].shape)
        start_shape, end_shape = (
            self.frame_shape[:stack_axis],
            self.frame_shape[stack_axis:],
        )
        self.shape = (
            *start_shape,
            end_shape[0] * len(frames),
            *end_shape[1:],
        )
        self.stack_axis = stack_axis
        self.dtype = frames[0].dtype
        if lz4_compress:
            frames = [compress(frame) for frame in frames]
        self._frames = frames
        self.lz4_compress = lz4_compress

    def __array__(self, dtype=None):
        arr = self[:]
        if dtype is not None:
            return arr.astype(dtype)
        return arr

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, int_or_slice):
        if isinstance(int_or_slice, int):
            return self._check_decompress(self._frames[int_or_slice])  # single frame
        return np.concatenate(
            [self._check_decompress(f) for f in self._frames[int_or_slice]],
            axis=self.stack_axis,
        )

    def __eq__(self, other):
        return self.__array__() == other

    def _check_decompress(self, frame):
        if self.lz4_compress:
            return np.frombuffer(decompress(frame), dtype=self.dtype).reshape(
                self.frame_shape
            )
        return frame


class LazyVecStackedObservations(list):
    """A simple list subclass, which supports conversion to a numpy array"""

    def __array__(self, dtype=None):
        arr = np.stack([obs.__array__() for obs in self])
        if dtype is not None:
            return arr.astype(dtype)
        return arr

    @property
    def shape(self):
        return (len(self), *self[0].shape)


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
        clone_arrays (bool): when using procgen with DummyVecEnv there may be some memory issues,
                             which can be fixed by copying data
        lz4_compress (bool, optional): use lz4 to compress the frames internally
        channel_axis (int, optional): channel axis for the observation data, along which successive frames are stacked.
                                      Defaults to -1.
    """

    def __init__(
        self,
        venv: VecEnv,
        num_stack: int,
        clone_arrays: bool,
        lz4_compress: bool = False,
        channel_axis: int = -1,
    ):
        super().__init__(venv)
        self.num_stack = num_stack
        self.lz4_compress = lz4_compress
        self.clone_arrays = clone_arrays
        self.stack_axis = channel_axis

        self.frames = [deque(maxlen=num_stack) for _ in range(self.num_envs)]

        low = np.repeat(self.observation_space.low, num_stack, axis=channel_axis)
        high = np.repeat(self.observation_space.high, num_stack, axis=channel_axis)
        self.observation_space = Box(
            low=low, high=high, dtype=self.observation_space.dtype
        )

    def _get_observation(self) -> LazyVecStackedObservations:
        result = []
        for i in range(len(self.frames)):
            assert len(self.frames[i]) == self.num_stack, (
                len(self.frames[i]),
                self.num_stack,
            )
            result.append(
                LazyStackedObservations(
                    list(self.frames[i]), self.lz4_compress, stack_axis=self.stack_axis
                )
            )
        return LazyVecStackedObservations(result)

    def step_wait(
        self,
    ) -> Tuple[LazyVecStackedObservations, np.ndarray, np.ndarray, List[Dict]]:
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

    def reset(self, **kwargs) -> LazyVecStackedObservations:
        observations = self.venv.reset(**kwargs)

        for i, observation in enumerate(observations):
            for _ in range(self.num_stack):
                self.frames[i].append(observation)
        return self._get_observation()

    def close(self) -> None:
        self.venv.close()

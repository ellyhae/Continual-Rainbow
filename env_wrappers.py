"""
This file is a slightly modified version of https://github.com/schmidtdominik/Rainbow/blob/298c93d3d9322440d3a22cf24045b57af9c83fde/common/env_wrappers.py
Specifically, stable baselines related code has been added and additional type annotations and documentation have been inserted

Note from original file:
Here all environment wrappers are defined and environments are created and configured.
Some of these wrappers are based on https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/common/atari_wrappers.py
"""

from functools import partial
from types import SimpleNamespace
from typing import Any, Tuple, Dict

import cv2
import numpy as np

import gym
from gym import Env
from gym.wrappers import TimeLimit

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    AtariWrapper,
)
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from stable_baselines3.common.vec_env.vec_transpose import VecTransposeImage

from vec_envs import DummyVecEnvNoFlatten, LazyVecFrameStack, SubprocVecEnvNoFlatten

cv2.ocl.setUseOpenCL(False)


class RetroEpisodicLifeEnv(gym.Wrapper):
    """
    Like the EpisodicLifeEnv above but for retro environments.
    This wrapper tries to detect whether the environment provides life information and is only active if it does.
    """

    def __init__(self, env: Env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        super().__init__(self, env)
        self.lives = 0
        self.was_real_done = True
        self.enabled = True

    def step(self, action: Any) -> Tuple[Any, float, bool, Dict[Any, Any]]:
        obs, reward, done, info = self.env.step(action)
        if not self.enabled:
            return obs, reward, done, info

        if self.enabled and "lives" not in info:
            self.enabled = False
            return obs, reward, done, info

        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = info["lives"]
        if self.lives > lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, *args, **kwargs) -> Any:
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if not self.enabled:
            return self.env.reset(*args, **kwargs)

        if self.was_real_done:
            obs = self.env.reset(*args, **kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, info = self.env.step(0)
            self.lives = info["lives"]
        return obs


class SkipFrameEnv(gym.Wrapper):
    """Return only every `skip`-th frame without maxing consecutive frames"""

    def __init__(self, env: Env, skip: int):
        """Return only every `skip`-th frame without maxing consecutive frames"""
        super().__init__(env)
        self._skip = skip

    def step(self, action: Any) -> Tuple[Any, float, bool, Dict[Any, Any]]:
        """Repeat action, and sum reward"""
        total_reward = 0.0
        actual_rewards = []

        done = False
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            actual_rewards.append(reward)
            if done:
                break
        info["actual_rewards"] = actual_rewards
        return obs, total_reward, done, info


class StochasticFrameSkip(gym.Wrapper):
    """
    Stochastic frame skipping wrapper, often used with gym-retro.
    """

    def __init__(self, env: Env, num_substeps: int, stickprob: float, seed: int):
        super().__init__(self, env)
        self.num_substeps = num_substeps
        self.stickprob = stickprob
        self.current_action = None
        self.rng = np.random.RandomState(seed)
        self.supports_want_render = hasattr(env, "supports_want_render")

    def reset(self, *args, **kwargs) -> Any:
        self.current_action = None
        return self.env.reset(*args, **kwargs)

    def step(self, action: Any) -> Tuple[Any, float, bool, Dict[Any, Any]]:
        done = False
        total_reward = 0
        actual_rewards = []
        for i in range(self.num_substeps):
            # First step after reset, use action
            if self.current_action is None:
                self.current_action = action
            # First substep, delay with probability=stickprob
            elif i == 0:
                if self.rng.random() > self.stickprob:
                    self.current_action = action
            # Second substep, new action definitely kicks in
            elif i == 1:
                self.current_action = action
            if self.supports_want_render and i < self.num_substeps - 1:
                obs, reward, done, info = self.env.step(
                    self.current_action, want_render=False
                )
            else:
                obs, reward, done, info = self.env.step(self.current_action)
            total_reward += reward
            actual_rewards.append(reward)

            if done:
                break

        info["actual_rewards"] = actual_rewards
        return obs, total_reward, done, info

    def seed(self, s) -> None:
        self.rng.seed(s)


class WarpFrame(gym.ObservationWrapper):
    """Re-scale frame observation and possibly convert to grayscale

    Very similar to stable baseline's WarpFrame, but with some additional features"""

    def __init__(
        self,
        env: Env,
        width: int,
        height: int,
        grayscale: bool = True,
        interp=cv2.INTER_AREA,
        dict_space_key=None,
    ):
        """
        Warp frames to 84x84 as done in the Nature paper and later work.
        If the environment uses dictionary observations, `dict_space_key` can be specified which indicates which
        observation should be warped.
        """
        super().__init__(env)
        self.interp = interp
        self._width = width
        self._height = height
        self._grayscale = grayscale
        self._key = dict_space_key
        if self._grayscale:
            num_colors = 1
        else:
            num_colors = 3

        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._height, self._width, num_colors),
            dtype=np.uint8,
        )
        if self._key is None:
            original_space = self.observation_space
            self.observation_space = new_space
        else:
            original_space = self.observation_space.spaces[self._key]
            self.observation_space.spaces[self._key] = new_space
        assert original_space.dtype == np.uint8 and len(original_space.shape) == 3

    def observation(self, obs: Any) -> Any:
        if self._key is None:
            frame = obs
        else:
            frame = obs[self._key]

        if self._grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if frame.shape[0] != self._height or frame.shape[1] != self._width:  # ds maybe
            frame = cv2.resize(
                frame, (self._width, self._height), interpolation=self.interp
            )
        if self._grayscale:
            frame = np.expand_dims(frame, -1)

        if self._key is None:
            obs = frame
        else:
            obs = obs.copy()
            obs[self._key] = frame
        return obs


class RandomizeStateOnReset(gym.Wrapper):
    """
    Wrapper for retro environments which loads a random new retro state (in games that provide multiple levels/modes) after each episode.
    """

    def __init__(self, env: Env, seed: int):
        super().__init__(env)

        import retro_utils

        self.init_states = retro_utils.get_init_states()[self.env.gamename]
        print(self.init_states)
        self.rng = np.random.RandomState(seed)
        if self.init_states:
            self.unwrapped.load_state(
                self.init_states[self.rng.randint(0, len(self.init_states))]
            )

    def reset(self, *args, **kwargs) -> Any:
        if len(self.init_states) > 1:
            next_state = self.init_states[self.rng.randint(0, len(self.init_states))]
            print(f"Loading state {next_state}")
            self.unwrapped.load_state(next_state)
        return self.env.reset(*args, **kwargs)


class DecorrEnvWrapper(gym.Wrapper):
    """When executing several environments in parallel (VecEnv), they (usually) all start at the same state, leading their
    episodes to be correlated with each other, which can be detrimental to the learning process.

    This wrapper takes some number decorr_steps random actions when the environment is first reset, after which it becomes inactive."""

    def __init__(self, env: Env, decorr_steps: int):
        super().__init__(env)
        self.decorr_steps = decorr_steps
        self.done = False

    def reset(self, *args, **kwargs) -> Any:
        """If this is the first reset of the environment then perform decorr_steps random actions, then turn inactive"""
        state = self.env.reset(*args, **kwargs)

        if not self.done:
            for i in range(int(self.decorr_steps)):
                state, _, done, _ = self.env.step(self.env.action_space.sample())
                if done:
                    state = self.env.reset(*args, **kwargs)
            self.done = True
        return state


def create_atari_env(
    config: SimpleNamespace, instance_seed: int, instance: int, decorr_steps: int | None
) -> Env:
    """Creates a gym atari environment and wraps it for DeepMind-style Atari"""
    env = gym.make(config.env_name.removeprefix("atari:") + "NoFrameskip-v4")
    env = TimeLimit(env, config.time_limit)

    # this has to be applied before the EpisodicLifeEnv (in the AtariWrapper) wrapper, as
    # the termination at the end of a life would otherwise mess with the monitoring results
    env = Monitor(env, allow_early_resets=True)

    assert config.resolution[0] == config.resolution[1]
    env = AtariWrapper(
        env,
        noop_max=30,
        frame_skip=config.frame_skip,
        screen_size=config.resolution[0],
        terminal_on_life_loss=True,
        clip_reward=True,
    )

    if decorr_steps is not None:
        env = DecorrEnvWrapper(env, decorr_steps)

    return env


def create_retro_env(
    config: SimpleNamespace, instance_seed: int, instance: int, decorr_steps: int | None
) -> Env:
    """Creates a retro environment and applies recommended wrappers."""

    # import retro only when needed, therefore making an optional package that does not need to be installed
    import retro
    from retro.examples.discretizer import Discretizer

    use_restricted_actions = retro.Actions.FILTERED
    if config.retro_action_patch == "discrete":
        use_restricted_actions = retro.Actions.DISCRETE

    randomize_state_on_reset = False
    retro_state = (
        retro.State.DEFAULT
        if (config.retro_state in ["default", "randomized"])
        else config.retro_state
    )
    if config.retro_state == "randomized":
        randomize_state_on_reset = True
    env = retro.make(
        config.env_name.removeprefix("retro:"),
        state=retro_state,
        use_restricted_actions=use_restricted_actions,
    )
    if (
        randomize_state_on_reset
    ):  # note: this might mess with any EpisodicLifeEnv-like wrappers!
        env = RandomizeStateOnReset(env, instance_seed)
    if config.retro_action_patch == "single_buttons":
        env = Discretizer(
            env, [[x] for x in env.unwrapped.buttons if x not in ("SELECT", "START")]
        )

    # removed RecorderWrapper

    env = TimeLimit(env, max_episode_steps=config.time_limit)
    if config.frame_skip > 1:
        env = StochasticFrameSkip(
            env,
            seed=instance_seed,
            num_substeps=config.frame_skip,
            stickprob=config.retro_stickyprob,
        )
    env = Monitor(env, allow_early_resets=True)
    env = RetroEpisodicLifeEnv(env)
    env = ClipRewardEnv(env)
    env = WarpFrame(
        env,
        width=config.resolution[1],
        height=config.resolution[0],
        grayscale=config.grayscale,
    )

    # removed RecorderWrapper

    if decorr_steps is not None:
        env = DecorrEnvWrapper(env, decorr_steps)
    return env


def create_procgen_env(
    config: SimpleNamespace, instance_seed: int, instance: int
) -> Env:
    """Creates a procgen environment and applies recommended wrappers."""
    procgen_args = {
        k[8:]: v for k, v in vars(config).items() if k.startswith("procgen_")
    }
    procgen_args["start_level"] += 300_000 * instance
    env = gym.make(f"procgen:procgen-{config.env_name.lower()[8:]}-v0", **procgen_args)

    # removed RecorderWrapper

    if config.frame_skip > 1:
        print("Frame skipping for procgen enabled!")
        env = SkipFrameEnv(env, config.frame_skip)

    env = Monitor(env, allow_early_resets=True)

    # Note: openai doesn't use reward clipping for procgen with ppo & rainbow (https://arxiv.org/pdf/1912.01588.pdf)
    # env = ClipRewardEnv(env)

    env = WarpFrame(
        env,
        width=config.resolution[1],
        height=config.resolution[0],
        grayscale=config.grayscale,
    )

    # removed RecorderWrapper

    return env


def create_env_instance(
    args: SimpleNamespace, instance: int, decorr_steps: int | None
) -> Env:
    """Create a single instance of the specified environment

    The prefixes 'retro:', 'atari:' and 'procgen:' can be used to apply the corresponding preprocessing wrappers.
    If no prefix is present, then the environment is directly loaded from gym and only the Monitor wrapper is applied

    Args:
        args (SimpleNamespace): Configuration parameters
        instance (int): used in conjunction with args.seed to create a custom seed for each environment
        decorr_steps (int | None): take some number decorr_steps random actions in the environment upon the first reset, to decorrelate parallel environments

    Returns:
        Env: The specified environment with all necessary wrappers applied
    """
    instance_seed = args.seed + instance
    decorr_steps = None if decorr_steps is None else decorr_steps * instance

    if args.env_name.startswith("retro:"):
        env = create_retro_env(args, instance_seed, instance, decorr_steps)
    elif args.env_name.startswith("atari:"):
        env = create_atari_env(args, instance_seed, instance, decorr_steps)
    elif args.env_name.startswith("procgen:"):
        env = create_procgen_env(args, instance_seed, instance)
    else:
        # if no env type is specified, assume that no processing is needed
        env = gym.make(args.env_name)
        env = Monitor(env)
        if decorr_steps is not None:
            env = DecorrEnvWrapper(env, decorr_steps)
    if not args.env_name.startswith("procgen:"):
        env.seed(instance_seed)
        env.action_space.seed(instance_seed)
        env.observation_space.seed(instance_seed)
    return env


def create_env(args: SimpleNamespace, decorr_steps: int | None = None) -> VecEnv:
    """Create a vectorized environment of the specified type, with args.parallel_envs environments as specified

    similar to stable baselines' make_vec_env, but with the instance/rank of each environment as an additional argument

    Args:
        args (SimpleNamespace): Configuration Namespace. Needs to include everything needed to instantiate an environment,
                                as well as the number of parallel_envs, the flag subproc_vecenv to switch between
                                SubprocVecEnvNoFlatten (true) and DummyVecEnvNoFlatten (false), as well as
                                frame_stack, the number of frames used for frame stacking
        decorr_steps (int | None, optional): _description_. Defaults to None.

    Returns:
        VecEnv: The specified VecEnv
    """
    env_fns = [
        partial(create_env_instance, args=args, instance=i, decorr_steps=decorr_steps)
        for i in range(args.parallel_envs)
    ]
    vec_env = SubprocVecEnvNoFlatten if args.subproc_vecenv else DummyVecEnvNoFlatten
    env = vec_env(env_fns)
    env = LazyVecFrameStack(
        env, args.frame_stack, clone_arrays=not args.subproc_vecenv, lz4_compress=False
    )
    # add a VecTransposeImage wrapper that is always skipped
    # otherwise, BaseAlgorithm may add it automatically, breaking many things
    env = VecTransposeImage(env, skip=True)
    return env

import random
from copy import deepcopy
from typing import Tuple, List

from omegaconf import OmegaConf, DictConfig
from tqdm.auto import trange

import wandb
from wandb.wandb_run import Run
from wandb.integration.sb3 import WandbCallback

from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    BaseCallback,
)
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from stable_baselines3.common.utils import configure_logger
from stable_baselines3.common.logger import Logger

from wandb_format import WandbOutputFormat
from rainbow import Rainbow
from env_wrappers import create_env
from logging_callbacks import CBPLogger, WeightLogger

from cbp import CBP

import numpy as np
import torch
from torch.optim import Adam


def set_random(seed: int):
    """Set the random seed for numpy, pytorch and random"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_mean_ep_length(args: DictConfig) -> float:
    """Run a few iterations of the environment with random actions to estimate the mean episode length"""
    dc_args = deepcopy(args)
    dc_args.parallel_envs = 12
    dc_args.subproc_vecenv = True
    dc_env = create_env(dc_args)
    dc_env.reset()

    # Decorrelate envs
    ep_lengths = []
    for _ in trange(args.time_limit // 4 + 100):
        _, _, _, infos = dc_env.step(
            [dc_env.action_space.sample() for _ in range(dc_args.parallel_envs)]
        )
        for info, _ in zip(infos, range(dc_args.parallel_envs)):
            if "episode" in info.keys():
                ep_lengths.append(info["episode"]["l"])
    dc_env.close()

    mean_length = sum(ep_lengths) / len(ep_lengths)
    return mean_length


def set_up_env(cfg: DictConfig) -> VecEnv:
    """Instatiates the specified environment"""

    decorr_steps = None
    if cfg.decorr and not cfg.env_name.startswith("procgen:"):
        decorr_steps = get_mean_ep_length(cfg) // cfg.parallel_envs

    env = create_env(cfg, decorr_steps=decorr_steps)

    return env


def initialize_model(cfg: DictConfig, env: VecEnv) -> Rainbow:
    """Instantiates the specified Rainbow Algorithm"""

    cfg = OmegaConf.to_container(cfg, resolve=True)

    # Using omegaconf open_dict and copy we could work directly on a DictConfig
    # and therefore remove the string label notation.
    # However, DictConfig only allows simple types, meaning the optimizer class could not be stored directly
    # This is an issue, as it is part of a a dictionary parameter passed to Rainbow, and therefore
    # a bit unwieldy to pass separately

    # calculate the exploration_fraction parameter
    if "eps_decay_frames" in cfg["settings"]:
        cfg["settings"]["exploration_fraction"] = cfg["settings"].pop(
            "eps_decay_frames"
        ) / (cfg["training_frames"] + env.num_envs)

    # choose the appropriate optimizer class
    if isinstance(cfg["settings"]["policy_kwargs"]["optimizer_class"], str):
        cfg["settings"]["policy_kwargs"]["optimizer_class"] = (
            Adam
            if cfg["settings"]["policy_kwargs"]["optimizer_class"] == "Adam"
            else CBP
        )

    return Rainbow(env=env, **cfg["settings"])


def initialize_logging(
    cfg: DictConfig,
) -> Tuple[Run, Logger, List[BaseCallback]]:
    """Initializes Weights and Biases logger as well as logging, checkpointing and evaluation callbacks

    Args:
        cfg (DictConfig): Configuration

    Returns:
        Tuple[Run, Logger, List[BaseCallback]]
    """
    BaseAlgorithm.set_logger
    run = wandb.init(
        config=OmegaConf.to_container(cfg, resolve=True), reinit=True, **cfg.wandb
    )

    logger = configure_logger(
        0,
        cfg.algorithm.tensorboard_log,
        cfg.name,
        True,
        extra_formats=[WandbOutputFormat],
    )

    callbacks = [
        WeightLogger(),
        CBPLogger(cfg.cbp_dir, cfg.cbp_log_freq),
        CheckpointCallback(save_freq=cfg.checkpoint_freq, save_path=cfg.checkpoint_dir),
        WandbCallback(gradient_save_freq=cfg.wandb_gradient_save_freq),
    ]

    if cfg.eval.evalcallback:
        eval_args = deepcopy(cfg.env)
        eval_args.seed += 100  # change seed to avoid data leakage if necessary
        eval_args.decorr = False
        eval_args.parallel_envs = cfg.eval.n_eval_episodes

        callbacks.append(
            EvalCallback(
                set_up_env(eval_args),
                n_eval_episodes=cfg.eval.n_eval_episodes,
                eval_freq=cfg.eval.eval_freq,
                log_path=cfg.eval.dir,
                best_model_save_path=cfg.eval.best_model_dir,
                verbose=0,
            )
        )

    return run, logger, callbacks


@torch.no_grad()
def save_evaluation(cfg: DictConfig, model: BaseAlgorithm) -> None:
    """Evaluate the given model and store the results in final_evaluation.npy"""

    eval_args = deepcopy(cfg)

    # change seed for evaluation to avoid train-test contamination
    eval_args.random.seed += 2 * eval_args.env.parallel_envs

    eval_args.env.decorr = False
    eval_args.env.parallel_envs = min(
        eval_args.eval.n_eval_episodes, eval_args.env.parallel_envs
    )
    model.policy.eval()

    env = set_up_env(eval_args.env)

    res = evaluate_policy(
        model, env, eval_args.eval.n_eval_episodes, return_episode_rewards=True
    )

    env.close()

    np.save("final_evaluation", res)

from copy import copy, deepcopy

import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy

from omegaconf import OmegaConf
from tqdm.auto import trange

from sb3_logger import configure_logger, WandbOutputFormat
from rainbow import Rainbow

from env_wrappers import create_env
from utils import CBPLogger, WeightLogger

from cbp import CBP
from torch.optim import Adam

import numpy as np
import torch
import random

def set_random(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
def get_mean_ep_length(args):
    """Run a few iterations of the environment and estimate the mean episode length"""
    dc_args = deepcopy(args)
    dc_args.parallel_envs = 12
    dc_args.subproc_vecenv = True
    dc_env = create_env(dc_args)
    dc_env.reset()

    # Decorrelate envs
    ep_lengths = []
    for frame in trange(args.time_limit//4+100):
        _, _, _, infos = dc_env.step([dc_env.action_space.sample() for x in range(dc_args.parallel_envs)])
        for info, j in zip(infos, range(dc_args.parallel_envs)):
            if 'episode' in info.keys(): ep_lengths.append(info['episode']['l'])
    dc_env.close()
    mean_length = sum(ep_lengths)/len(ep_lengths)
    return mean_length

def set_up_env(cfg):
    decorr_steps = None
    if cfg.decorr and not cfg.env_name.startswith('procgen:'):
        decorr_steps = get_mean_ep_length(cfg) // cfg.parallel_envs
    env = create_env(cfg, decorr_steps=decorr_steps)
    
    return env

def initialize_model(cfg, env):
    cfg = OmegaConf.to_container(cfg, resolve=True)
    if 'eps_decay_frames' in cfg['settings']:
        cfg['settings']['exploration_fraction'] = cfg['settings'].pop('eps_decay_frames') / (cfg['training_frames'] + env.parallel_envs)
    if isinstance(cfg['settings']['policy_kwargs']['optimizer_class'], str):
        cfg['settings']['policy_kwargs']['optimizer_class'] = Adam if cfg['settings']['policy_kwargs']['optimizer_class'] == 'Adam' else CBP
    return Rainbow(env=env, **cfg['settings'])

def initialize_logging(cfg, model):
    run = wandb.init(
        config=OmegaConf.to_container(cfg, resolve=True),
        reinit=True,
        **cfg.wandb
    )
    
    model.set_logger(configure_logger(0, cfg.algorithm.tensorboard_log, cfg.name, True, extra_formats=[WandbOutputFormat]))
    
    callbacks = [WeightLogger(),
                 CBPLogger(cfg.cbp_dir, cfg.cbp_log_freq),
                 WandbCallback(gradient_save_freq=cfg.wandb_gradient_save_freq)]
    
    if cfg.eval.evalcallback:
        eval_args = copy(cfg.env)
        eval_args.seed += 1   # change seed to avoid data leakage if necessary
        eval_args.decorr = False
        eval_args.parallel_envs = cfg.eval.n_eval_episodes
        
        callbacks.append(EvalCallback(set_up_env(eval_args),
                                      n_eval_episodes=cfg.eval.n_eval_episodes,
                                      eval_freq=cfg.eval.eval_freq,
                                      log_path=cfg.eval.dir,
                                      best_model_save_path=cfg.eval.best_model_dir,
                                      verbose=0))
    
    return run, callbacks

@torch.no_grad()
def save_evaluation(cfg, model):
    cfg.random.seed += 2 * cfg.env.parallel_envs # change seed for evaluation to avoid train-test contamination
    cfg.env.decorr = False
    cfg.env.parallel_envs = min(cfg.eval.n_eval_episodes, cfg.env.parallel_envs)
    model.policy.eval()
    
    env = set_up_env(cfg.env)
    
    res = evaluate_policy(model, env, cfg.eval.n_eval_episodes, return_episode_rewards=True)
    
    env.close()
    
    np.save('final_evaluation', res)
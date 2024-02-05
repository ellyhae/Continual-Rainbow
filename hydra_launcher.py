# python hydra_launcher.py --multirun algorithm=glob(*) random=glob(*)

import os

from omegaconf import DictConfig, OmegaConf
import hydra

import wandb

from run_utils import set_random, set_up_env, initialize_model, initialize_logging, save_evaluation

@hydra.main(version_base=None, config_path="hydra_config", config_name="config")
def run_experiment(cfg: DictConfig) -> None:
    set_random(cfg.random.seed)
    
    env = set_up_env(cfg.env)
    model = initialize_model(cfg.algorithm, env)
    run, callbacks = initialize_logging(cfg, model)
    
    model.learn(cfg.algorithm.training_frames, callback=callbacks, tb_log_name=cfg.name, progress_bar=False)
    model.save(os.path.join(cfg.model_dir, 'final_model'))
    
    env.close()
    
    save_evaluation(cfg, model)
    
    model.logger.close()
    run.finish()

if __name__ == "__main__":
    wandb.setup()
    run_experiment()
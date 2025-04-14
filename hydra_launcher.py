import os

import hydra
import wandb
from omegaconf import DictConfig

from run_utils import (
    initialize_logging,
    initialize_model,
    save_evaluation,
    set_random,
    set_up_env,
)


@hydra.main(version_base=None, config_path="hydra_config", config_name="config")
def run_experiment(cfg: DictConfig) -> None:
    """Entry point for experiments started with Hydra.
    The specified environment and model asre instatiated for the given seed, after which the training loop is executed.
    After training, the model is evaluated one last time.
    This function is executed for each configuration specified in the Hydra call.

    An example Hydra call is:
    python hydra_launcher.py --multirun algorithm=glob(*) random=glob(*)

    Args:
        cfg (DictConfig): The current configuration as provided by Hydra
    """
    set_random(cfg.random.seed)

    env = set_up_env(cfg.env)
    model = initialize_model(cfg.algorithm, env)
    run, callbacks = initialize_logging(cfg, model)

    model.learn(
        cfg.algorithm.training_frames,
        callback=callbacks,
        tb_log_name=cfg.name,
        progress_bar=cfg.progress_bar,
    )
    model.save(os.path.join(cfg.model_dir, "final_model"))

    # Close any processes or files opened by the environment, as they are no longer needed
    env.close()

    save_evaluation(cfg, model)

    # to prevent errors, close the logger before finishing a run, thereby ensuring that all data has been processed
    model.logger.close()
    run.finish()


if __name__ == "__main__":
    wandb.setup()
    run_experiment()

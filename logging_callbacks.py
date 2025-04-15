# TODO add/fix docstrings
# TODO add comments

import os

import numpy as np
import torch

from stable_baselines3.common.callbacks import BaseCallback


class WeightLogger(BaseCallback):
    def _on_step(self) -> bool:
        return True

    def _on_rollout_start(self) -> None:
        self._log()

    def _on_training_end(self) -> None:
        self._log()

    def _log(self) -> None:
        params = np.abs(self.model.q_net.parameters_to_vector())
        self.logger.record("train/weight_magnitude_mean", float(params.mean()))
        self.logger.record("train/weight_magnitude_std", float(params.std()))
        self.logger.dump(self.num_timesteps)


class CBPLogger(BaseCallback):
    def __init__(
        self, save_dir: str, logging_freq: int = 500, verbose: int = 0
    ) -> None:
        """
        save_dir: path to directory where values should be stored
        logging_freq: how often to log values other than ages (ages are always logged when changed)
        """
        super().__init__(verbose)
        self.save_dir = save_dir
        self.logging_freq = logging_freq
        self.iteration = 0
        self.end = False

    def _init_callback(self) -> None:
        os.makedirs(self.save_dir, exist_ok=True)
        self.active = hasattr(self.model.policy.optimizer, "cbp_vals")
        if self.active:
            self.ages = {}
            for branch, branch_name in zip(
                self.model.policy.optimizer.linear_layers, ["value", "advantage"]
            ):
                for i, layer in enumerate(branch[:-1]):
                    self.ages[f"{branch_name}_{i}"] = (
                        self.model.policy.optimizer.cbp_vals[layer]["age"]
                        .detach()
                        .clone()
                    )

    def _on_rollout_start(self) -> None:
        self._save()

    def _on_training_end(self) -> None:
        self.end = True
        self._save()

    def _on_step(self) -> bool:
        return True

    @torch.no_grad()
    def _save(self) -> None:
        if self.active and self.model.replay_buffer.size() > self.model.learning_starts:
            logged = False
            age_logged = False
            opt = self.model.policy.optimizer
            cbp_values = {}
            for branch, branch_name in zip(opt.linear_layers, ["value", "advantage"]):
                for i, layer in enumerate(branch[:-1]):
                    self.ages[f"{branch_name}_{i}"].add_(
                        self.model.gradient_steps
                    )  # simulate updates without resets
                    cbp_vals = opt.cbp_vals[layer]
                    val = cbp_vals["age"]
                    if (self.ages[f"{branch_name}_{i}"] != val).any() or self.end:
                        self.ages[f"{branch_name}_{i}"].copy_(val)
                        self.logger.record(
                            f"age/{branch_name}_{i}",
                            val.cpu().numpy(),
                            exclude="tensorboard",
                        )
                        age_logged = True
                    if self.iteration % self.logging_freq == 0 or self.end:
                        for name, val in cbp_vals.items():
                            self.logger.record(
                                f"{name}/{branch_name}_{i}",
                                val.cpu().numpy(),
                                exclude="tensorboard",
                            )
                            cbp_values[f"{name}/{branch_name}_{i}"] = val
                        logged = True

            if logged or age_logged:
                self.logger.dump(self.num_timesteps)
            if logged:
                torch.save(
                    self.ages,
                    os.path.join(self.save_dir, str(self.iteration) + "_vals.pt"),
                )
            if age_logged:
                torch.save(
                    self.ages,
                    os.path.join(self.save_dir, str(self.iteration) + "_age.pt"),
                )

                if len(opt.state) > 0:
                    # gather step arrays into dictionary with appropriate keys and save
                    steps = {}
                    for branch, branch_name in zip(
                        opt.linear_layers, ["value", "advantage"]
                    ):
                        for i, layer in enumerate(branch):
                            steps[f"{branch_name}_{i}"] = opt.state[layer.weight][
                                "step"
                            ]
                    torch.save(
                        steps,
                        os.path.join(self.save_dir, str(self.iteration) + "_step.pt"),
                    )

            self.iteration += 1

import os

import numpy as np
import torch

from stable_baselines3.common.callbacks import BaseCallback

from cbp import CBP


class WeightLogger(BaseCallback):
    """Log the mean and std of the weight magnitudes for the online q-network before every rollout"""

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
    """Log CBP values periodically, with the unit age bein logged whenever it changes.

    Data is stored to disk for later analysis, see _save

    If CBP is not used as the policy optimizer, then this is a No-Op"""

    opt: CBP

    def __init__(
        self, save_dir: str, logging_freq: int = 500, verbose: int = 0
    ) -> None:
        """
        Args:
            save_dir (str): path to directory where ages and values should be stored
            logging_freq (int, optional): how often to log values other than ages
                (ages are always logged when changed). Defaults to 500.
            verbose (int, optional): Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
                debug messages. Defaults to 0.
        """
        super().__init__(verbose)
        self.save_dir = save_dir
        self.logging_freq = logging_freq
        self.iteration = 0
        self.end = False

    def _init_callback(self) -> None:
        # determine if the optimizer is CBP
        self.active = isinstance(self.model.policy.optimizer, CBP)
        if self.active:
            os.makedirs(self.save_dir, exist_ok=True)

            self.opt = self.model.policy.optimizer

            # prepare a dictionary similar to the optimizer state, which will be
            # used to keep track of age changes
            self.ages = {}
            for branch, branch_name in zip(
                self.opt.linear_layers, ["value", "advantage"]
            ):
                for i, layer in enumerate(branch[:-1]):
                    self.ages[f"{branch_name}_{i}"] = (
                        self.opt.cbp_vals[layer]["age"].detach().clone()
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
        """Log age information if it has changed, log other CBP values periodically.

        The data is also saved to disk whenever something is logged, with the number in the
        file name indicating the n-th step in the learning process. The _vals suffix
        indicates files with all cbp values, while _age only contains age data, but is logged after every change.
        Finally, _step is the per scalar weight age used by CAdam and accompanies _age files
        """
        if self.active and self.model.replay_buffer.size() > self.model.learning_starts:
            logged = False
            age_logged = False
            opt: CBP = self.model.policy.optimizer
            cbp_values = {}

            # do the logging for both branches and all layers of the online q-network
            for branch, branch_name in zip(opt.linear_layers, ["value", "advantage"]):
                for i, layer in enumerate(branch[:-1]):
                    # simulate age updates without resets
                    self.ages[f"{branch_name}_{i}"].add_(self.model.gradient_steps)
                    # get actual values
                    cbp_vals = opt.cbp_vals[layer]
                    cbp_age = cbp_vals["age"]

                    # if any age entry differs from the simulation, then there was reset and it needs to be logged
                    if (self.ages[f"{branch_name}_{i}"] != cbp_age).any() or self.end:
                        self.ages[f"{branch_name}_{i}"].copy_(cbp_age)
                        self.logger.record(
                            f"age/{branch_name}_{i}",
                            cbp_age.cpu().numpy(),
                            exclude="tensorboard",
                        )
                        age_logged = True

                    # periodically record all values that are being tracked by CBP
                    if self.iteration % self.logging_freq == 0 or self.end:
                        for name, value in cbp_vals.items():
                            self.logger.record(
                                f"{name}/{branch_name}_{i}",
                                value.cpu().numpy(),
                                exclude="tensorboard",
                            )
                            cbp_values[f"{name}/{branch_name}_{i}"] = value
                        logged = True

            if logged or age_logged:
                self.logger.dump(self.num_timesteps)

            # if something was logged, also save everything to disk for later analysis
            if logged:
                torch.save(
                    cbp_values,
                    os.path.join(self.save_dir, str(self.iteration) + "_vals.pt"),
                )
            if age_logged:
                torch.save(
                    self.ages,
                    os.path.join(self.save_dir, str(self.iteration) + "_age.pt"),
                )

                # if the optimizer has populated the state dictionary, also save the optimizer's default step
                # Note: the difference between CBP age and Adam step is that CBP works on a per unit basis,
                # while Adam works on a per scalar basis. In principle, the 2D age matrix of Adam could most likely be
                # calculated using the 1D CBP ages, but storing them seems to save some computation each update
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

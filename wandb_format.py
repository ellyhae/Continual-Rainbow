"""
Defines a custom Weights and Biases logger output format based on stable baseline's common/logger.py
"""

from typing import Any, Dict, Tuple, Union

import numpy as np
import wandb
from torch import Tensor
from PIL import Image as PILImage

from stable_baselines3.common.logger import (
    Figure,
    HParam,
    Image,
    KVWriter,
    Video,
)


class WandbOutputFormat(KVWriter):
    """
    Dumps key/value pairs into Weights & Biases.

    Adapted from stable baseline's TensorBoardOutputFormat
    """

    def __init__(self, log_dir: str = "", log_suffix: str = ""):
        """Initialize the output format

        The Weights and Biases run to which values are to be logged should be initialized before creating this object.

        :param log_dir: Ignored. The weights and bisases run already specifies the output directory
        :param: log_suffix: Ignored. The weights and bisases run already specifies the output name
        :raises wandb.Error: wandb.init must used to instantiate a Weights and Biases run before creating this object
        """
        if wandb.run is None:
            raise wandb.Error("You must call wandb.init() before WandbOutputFormat()")

        wandb.define_metric("timestep")
        wandb.define_metric("*", step_metric="timestep")

    def write(
        self,
        key_values: Dict[str, Any],
        key_excluded: Dict[str, Union[str, Tuple[str, ...]]],
        step: int = 0,
    ) -> None:
        """
        Write a dictionary to file

        :param key_values:
        :param key_excluded:
        :param step: step in the learning process for which to log the given data
        """

        log_dict = {}

        for (key, value), (_, excluded) in zip(
            sorted(key_values.items()), sorted(key_excluded.items())
        ):
            if excluded is not None and "wandb" in excluded:
                continue

            if isinstance(value, (np.ScalarType, dict, list, np.ndarray, Tensor)):
                log_dict[key] = value

            if isinstance(value, Video):
                log_dict[key] = wandb.Video(value.frames, fps=value.fps)

            if isinstance(value, Figure):
                log_dict[key] = wandb.Image(value.figure)
                if value.close:
                    value.figure.close()

            if isinstance(value, Image):
                log_dict[key] = wandb.Image(PILImage.fromarray(value.image, mode="RGB"))

            if isinstance(value, HParam):
                wandb.config.update(value.hparam_dict)
                for k, v in value.metric_dict:
                    wandb.run.summary[k] = v

        log_dict["timestep"] = step

        # Flush the output to the file
        wandb.log(log_dict)

    def close(self) -> None:
        """
        finishes the wandb run
        """
        wandb.finish()

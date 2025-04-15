# TODO add/fix docstrings
# TODO add comments

from typing import Dict, List, Type, Any, Optional, Callable
from math import sqrt

import gym
from gym import spaces

import torch
from torch import nn as nn, Tensor
from torch.nn import init
import torch.nn.functional as F

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# torch.backends.cudnn.deterministic = True


# adapted from Rainbow/common/networks.py
class FactorizedNoisyLinear(nn.Module):
    """The factorized Gaussian noise layer for noisy-nets dqn."""

    def __init__(
        self, in_features: int, out_features: int, sigma_0: float = 0.5
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_0 = sigma_0

        # weight: w = \mu^w + \sigma^w . \epsilon^w
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))

        # bias: b = \mu^b + \sigma^b . \epsilon^b
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        # initialization is similar to Kaiming uniform (He. initialization) with fan_mode=fan_in
        scale = 1 / sqrt(self.in_features)

        init.uniform_(self.weight_mu, -scale, scale)
        init.uniform_(self.bias_mu, -scale, scale)

        init.constant_(self.weight_sigma, self.sigma_0 * scale)
        init.constant_(self.bias_sigma, self.sigma_0 * scale)

    @torch.no_grad()
    def _get_noise(self, size: int) -> Tensor:
        noise = torch.randn(size, device=self.weight_mu.device)
        # f(x) = sgn(x)sqrt(|x|)
        return noise.sign().mul_(noise.abs().sqrt_())

    @torch.no_grad()
    def reset_noise(self) -> None:
        # like in eq 10 and 11 of the paper
        epsilon_in = self._get_noise(self.in_features)
        epsilon_out = self._get_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    @torch.no_grad()
    def disable_noise(self) -> None:
        self.weight_epsilon[:] = 0
        self.bias_epsilon[:] = 0

    def forward(self, input: Tensor) -> Tensor:
        # y = wx + d, where
        # w = \mu^w + \sigma^w * \epsilon^w
        # b = \mu^b + \sigma^b * \epsilon^b

        # save in class variable for external access
        self.weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
        self.bias = self.bias_mu + self.bias_sigma * self.bias_epsilon

        return F.linear(input, self.weight, self.bias)


# adapted from stable_baselines3/common/torch_layers.py
def create_mlp(
    input_dim: int,
    output_dim: int,
    net_arch: List[int],
    activation_fn: Type[nn.Module] = nn.ReLU,
    noisy_linear: bool = False,
    linear_kwargs: Dict[str, Any] = {},
    squash_output: bool = False,
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.
    Optionally: use noisy Linear layers instead of normal Linear layers (paper NOISY NETWORKS FOR EXPLORATION)

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :param noisy_linear: If set to True, noisy Linear layers will be used instead of normal Linear layers
    :param linear_kwargs: Keyword arguments for the linear layers
    :return:
    """

    linear_layer = nn.Linear if not noisy_linear else FactorizedNoisyLinear

    if len(net_arch) > 0:
        modules = [
            linear_layer(input_dim, net_arch[0], **linear_kwargs),
            activation_fn(),
        ]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(linear_layer(net_arch[idx], net_arch[idx + 1], **linear_kwargs))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(linear_layer(last_layer_dim, output_dim, **linear_kwargs))
    if squash_output:
        modules.append(nn.Tanh())
    return modules


# adapted from Rainbow/common/networks.py
class Dueling(nn.Module):
    """The dueling branch used in all nets that use dueling-dqn."""

    def __init__(self, value_branch: nn.Module, advantage_branch: nn.Module) -> None:
        super().__init__()
        self.value_branch = value_branch
        self.advantage_branch = advantage_branch

    def forward(self, x: Tensor, advantages_only: bool = False) -> Tensor:
        advantages = self.advantage_branch(x)
        if advantages_only:
            return advantages

        value = self.value_branch(x)
        return value + (advantages - torch.mean(advantages, dim=1, keepdim=True))


# copied from Rainbow/common/networks.py
class ImpalaCNNResidual(nn.Module):
    """
    Simple residual block used in the large IMPALA CNN.
    """

    def __init__(self, depth: int, norm_func: Callable) -> None:
        super().__init__()

        self.relu = nn.ReLU()
        self.conv_0 = norm_func(
            nn.Conv2d(
                in_channels=depth,
                out_channels=depth,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        )
        self.conv_1 = norm_func(
            nn.Conv2d(
                in_channels=depth,
                out_channels=depth,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        x_ = self.conv_0(self.relu(x))
        x_ = self.conv_1(self.relu(x_))
        return x + x_


# copied from Rainbow/common/networks.py
class ImpalaCNNBlock(nn.Module):
    """
    Three of these blocks are used in the large IMPALA CNN.
    """

    def __init__(self, depth_in: int, depth_out: int, norm_func: Callable) -> None:
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=depth_in,
            out_channels=depth_out,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.max_pool = nn.MaxPool2d(3, 2, padding=1)
        self.residual_0 = ImpalaCNNResidual(depth_out, norm_func=norm_func)
        self.residual_1 = ImpalaCNNResidual(depth_out, norm_func=norm_func)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.max_pool(x)
        x = self.residual_0(x)
        x = self.residual_1(x)
        return x


class ImpalaCNNLarge(BaseFeaturesExtractor):
    """
    CNN part of the Impala Architecture plus a single (noisy) Linear layer. Extracts features from images
    :param observation_space:
    :param features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    :param normalized_image: Whether to assume that the image is already normalized
        or not (this disables dtype and bounds checks): when True, it only checks that
        the space is a Box and has 3 dimensions.
        Otherwise, it checks that it has expected dtype (uint8) and bounds (values in [0, 255]).
    """

    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: Optional[int] = None,  # only flatten unless value is supplied
        model_size: int = 2,
        spectral_norm: str = "all",
        activation_fn: Type[nn.Module] = nn.ReLU,
        # normalized_image: bool = False,
    ) -> None:
        assert isinstance(observation_space, spaces.Box), (
            "ImpalaCNNLarge must be used with a gym.spaces.Box ",
            f"observation space, not {observation_space}",
        )
        super().__init__(observation_space, 1)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        # assert is_image_space(observation_space, check_channels=False), ( #, normalized_image=normalized_image
        #    "You should use ImpalaCNNLarge "
        #    f"only with images not with {observation_space}\n"
        #    "(you are probably using `CnnPolicy` instead of `MlpPolicy` or `MultiInputPolicy`)\n"
        #    "If you are using a custom environment,\n"
        #    "please check it using our env checker:\n"
        #    "https://stable-baselines3.readthedocs.io/en/master/common/env_checker.html.\n"
        #    "If you are using `VecNormalize` or already normalized channel-first images "
        #    "you should pass `normalize_images=False`: \n"
        #    "https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html"
        # )
        n_input_channels = observation_space.shape[0]

        norm_func = (
            torch.nn.utils.spectral_norm if (spectral_norm == "all") else nn.Identity()
        )
        norm_func_last = (
            torch.nn.utils.spectral_norm
            if (spectral_norm == "last" or spectral_norm == "all")
            else nn.Identity()
        )

        self.main = torch.nn.Sequential(
            ImpalaCNNBlock(n_input_channels, 16 * model_size, norm_func=norm_func),
            ImpalaCNNBlock(16 * model_size, 32 * model_size, norm_func=norm_func),
            ImpalaCNNBlock(32 * model_size, 32 * model_size, norm_func=norm_func_last),
            activation_fn(),
            torch.nn.AdaptiveMaxPool2d(
                (8, 8)
            ),  # reduces output to shape (batch_size, 32*model_size, 8, 8)
            torch.nn.Flatten(),
        )

        n_flatten = 32 * model_size * 8**2

        self.last = nn.Identity()
        if features_dim is not None:
            self.last = nn.Sequential(
                torch.nn.Linear(n_flatten, features_dim), torch.nn.ReLU()
            )
            self._features_dim = features_dim
        else:
            self._features_dim = n_flatten

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.last(self.main(observations))

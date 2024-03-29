from typing import Any, Dict, List, Optional, Type

import torch as torch
from gym import spaces
from torch import nn
import numpy as np

from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    #NatureCNN,
    #create_mlp,
)
from stable_baselines3.common.type_aliases import Schedule

from torch_layers import (
    ImpalaCNNLarge,
    create_mlp,
    Dueling
)

from utils import prep_observation_for_qnet

from cbp import CBP, prepare_cbp_kwargs

class RainbowNetwork(BasePolicy):
    """
    network for Rainbow
    :param observation_space: Observation space
    :param action_space: Action space
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """

    action_space: spaces.Discrete

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        features_extractor: BaseFeaturesExtractor,
        features_dim: Optional[int] = None,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        noisy_linear: bool = True,
        linear_kwargs: Dict[str, Any] = {'sigma_0': 0.5}
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        self.contructor_features_dim = features_dim

        if features_dim is None:
            features_dim = self.features_extractor.features_dim

        if net_arch is None:
            net_arch = [256]

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.features_dim = features_dim
        action_dim = int(self.action_space.n)  # number of actions
        self.noisy_linear = noisy_linear
        self.linear_kwargs = linear_kwargs
        self.dueling = Dueling(
            nn.Sequential(*create_mlp(self.features_dim, 1, self.net_arch, self.activation_fn, self.noisy_linear, self.linear_kwargs)),
            nn.Sequential(*create_mlp(self.features_dim, action_dim, self.net_arch, self.activation_fn, self.noisy_linear, self.linear_kwargs))
        )

    def forward(self, obs: torch.Tensor, advantages_only: bool = False) -> torch.Tensor:
        """
        Predict the q-values.
        :param obs: Observation
        :return: The estimated Q-Value for each action.
        """
        return self.dueling(self.extract_features(obs), advantages_only=advantages_only)

    def _predict(self, observation: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        q_values = self(observation, advantages_only=True)
        # Greedy action
        action = q_values.argmax(dim=1).reshape(-1)
        return action

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.contructor_features_dim,
                activation_fn=self.activation_fn,
                features_extractor=self.features_extractor,
                noisy_linear=self.noisy_linear,
                linear_kwargs=self.linear_kwargs
            )
        )
        return data
    
    #def set_training_mode(self, mode: bool) -> None:
    #    self.training_ = mode
    #
    #def is_training(self) -> bool:
    #    return self.training_


class RainbowPolicy(BasePolicy):
    """
    Policy class with Q-Value Net and target net for DQN
    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    q_net: RainbowNetwork
    q_net_target: RainbowNetwork

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        noisy_linear: bool = True,
        linear_kwargs: Dict[str, Any] = {'sigma_0': 0.5},
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        use_amp: bool = True
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            normalize_images=normalize_images,
        )

        if net_arch is None:
            if features_extractor_class == ImpalaCNNLarge:
                net_arch = [256]
            else:
                net_arch = [64, 64]

        self.net_arch = net_arch
        self.activation_fn = activation_fn

        self.noisy_linear = noisy_linear
        self.linear_kwargs = linear_kwargs

        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "net_arch": self.net_arch,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
            "noisy_linear": self.noisy_linear,
            "linear_kwargs": linear_kwargs,
        }
        
        self.use_amp = use_amp

        self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the network and the optimizer.
        Put the target network into evaluation mode.
        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """

        self.q_net = self.make_net()
        self.q_net_target = self.make_net()
        self.q_net_target.load_state_dict(self.q_net.state_dict())
        
        # Cant set the target's training mode to false, as this affects the results of spectral norm and causes the results to differ from the reference implementation
        self.q_net_target.set_training_mode(False)

        if self.optimizer_class == CBP:
            optimizer_kwargs = self.optimizer_kwargs | prepare_cbp_kwargs(self.q_net, self.net_args)
        else:
            optimizer_kwargs = self.optimizer_kwargs
        
        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(  # type: ignore[call-arg]
            self.q_net.parameters(),
            lr=lr_schedule(1),
            **optimizer_kwargs,
        )

    def make_net(self) -> RainbowNetwork:
        # Make sure we always have separate networks for features extractors etc
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return RainbowNetwork(**net_args).to(self.device)

    def forward(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        return self._predict(obs, deterministic=deterministic)

    def _predict(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        return self.q_net._predict(obs, deterministic=deterministic)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_args["net_arch"],
                activation_fn=self.net_args["activation_fn"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
            )
        )
        return data
    
    def obs_to_tensor(self, observation):
        """
        Overwrites default obs_to_tensor function.
        Loses flexibility (non-vectorized envs, ...)
        Adds frame stacking support
        """
        if isinstance(observation, list):
            return prep_observation_for_qnet(torch.from_numpy(np.stack(observation)), self.use_amp), True
        return super().obs_to_tensor(observation)

    #def set_training_mode(self, mode: bool) -> None:
    #    """
    #    As the reference implementation does not distinguish between train and eval modes, neither will we
    #    
    #    Put the policy in either training or evaluation mode.
    #    This affects certain modules, such as batch normalisation and dropout.
    #    :param mode: if true, set to training mode, else set to evaluation mode
    #    """
    #    self.q_net.set_training_mode(mode)
    #    #self.training = mode


MlpPolicy = RainbowPolicy


class CnnPolicy(RainbowPolicy):
    """
    Policy class for DQN when using images as input.
    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param features_extractor_class: Features extractor to use.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = ImpalaCNNLarge,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        noisy_linear: bool = True,
        linear_kwargs: Dict[str, Any] = {'sigma_0': 0.5},
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            features_extractor_class,
            features_extractor_kwargs,
            normalize_images,
            noisy_linear,
            linear_kwargs,
            optimizer_class,
            optimizer_kwargs,
        )
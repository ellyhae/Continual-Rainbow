# TODO add/fix docstrings
# TODO add comments

from typing import Any, Dict, List, Optional, Type, Tuple

import torch as torch
from torch import nn
from gym import spaces

from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
)
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.preprocessing import preprocess_obs

from vec_envs import LazyVecStackedObservations
from torch_layers import ImpalaCNNLarge, create_mlp, Dueling
from cbp import CBP, prepare_cbp_kwargs


class RainbowNetwork(BasePolicy):
    """A Q-Net for the Rainbow algorithm, using dueling networks with optional noisy linears"""

    action_space: spaces.Discrete

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        force_normalize_obs: bool = True,
        noisy_linear: bool = True,
        linear_kwargs: Dict[str, Any] | None = None,
        use_amp: bool = True,
    ) -> None:
        """
        Args:
            observation_space (spaces.Space): Observation space of the environment
            action_space (spaces.Discrete): Action space of the environment
            features_extractor (BaseFeaturesExtractor): Module used to convert input observations to flattened feature tensor
            features_dim (int): length of the flattened extracted features
            net_arch (List[int]): Architecture of the multi layer perceptron (MLP).
                It represents the number of units per layer. The length of this list is the number of layers.
            activation_fn (Type[nn.Module], optional): Activation function for MLP. Defaults to nn.ReLU.
            normalize_images (bool, optional): Whether to normalize images or not, dividing by 255.0. Defaults to True.
            force_normalize_obs (bool, optional): Due to frame stacking stable baselines may not be able to recognize image data, therefore skipping normalization and channel reordering.
                If this is set to True, the usual image preprocessing is always done, no matter the observation space. Defaults to True.
            noisy_linear (bool, optional): Use noisy linear layers in the MLP. Defaults to True.
            linear_kwargs (Dict[str, Any] | None, optional): Additional arguments for the (noisy) linear layers in the MLP. Defaults to None.
            use_amp (bool, optional): Use Pytorch Automatic Mixed Precision, converting observations to float16. Defaults to True.
        """
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        if linear_kwargs is None:
            linear_kwargs = {}

        self.force_normalize_obs = force_normalize_obs
        self.use_amp = use_amp
        self.amp_dtype = torch.float16 if self.use_amp else torch.float32

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.features_dim = features_dim
        action_dim = int(self.action_space.n)  # number of actions
        self.noisy_linear = noisy_linear
        self.linear_kwargs = linear_kwargs
        self.dueling = Dueling(
            nn.Sequential(
                *create_mlp(
                    self.features_dim,
                    1,
                    self.net_arch,
                    self.activation_fn,
                    self.noisy_linear,
                    self.linear_kwargs,
                )
            ),
            nn.Sequential(
                *create_mlp(
                    self.features_dim,
                    action_dim,
                    self.net_arch,
                    self.activation_fn,
                    self.noisy_linear,
                    self.linear_kwargs,
                )
            ),
        )

    def forward(self, obs: torch.Tensor, advantages_only: bool = False) -> torch.Tensor:
        """Estimate the Q-values for each action

        Args:
            obs (torch.Tensor): observations tensor
            advantages_only (bool, optional): Only use the advantages branch of the dueling network. Defaults to False.

        Returns:
            torch.Tensor: The estimated Q-Value for each action.
        """
        return self.dueling(self.extract_features(obs), advantages_only=advantages_only)

    def extract_features(self, obs: torch.Tensor) -> torch.Tensor:
        """Preprocess the observation if needed and extract features.

        Args:
            obs (torch.Tensor): observations tensor

        Returns:
            torch.Tensor: features tensor
        """
        assert self.features_extractor is not None, "No features extractor was set"

        preprocessed_obs = self.preprocess_obs(obs)

        return self.features_extractor(preprocessed_obs)

    def preprocess_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Preprocess observations such that the features extractor can use them.

        Extends the stable baselines preprocessed_obs util by allowing the user to force
        image normalization. Can be necessary when e.g. frame stacking produces observations that
        stable baselines does not recognize as images.

        Args:
            obs (torch.Tensor): observations tensor

        Returns:
            torch.Tensor: preprocessed observations
        """
        if self.force_normalize_obs:
            preprocessed_obs = obs.permute(
                (2, 0, 1) if len(obs.shape) == 3 else (0, 3, 1, 2)
            )
            return preprocessed_obs.to(dtype=self.amp_dtype) / 255

        preprocessed_obs = preprocess_obs(
            obs, self.observation_space, self.normalize_images
        )
        return preprocessed_obs.to(dtype=self.amp_dtype)

    def _predict(
        self, observation: torch.Tensor, deterministic: bool = False
    ) -> torch.Tensor:
        """Get the greedy action according to the policy for a given observation.

        Pretty much the exact same as in stable baseline's DQN

        Args:
            observation (torch.Tensor): observation Tensor
            deterministic (bool, optional): Ignored. If the network is deterministic, then so is this function.

        Returns:
            torch.Tensor: Taken action according to the policy
        """

        q_values = self(observation, advantages_only=True)
        # Greedy action
        action = q_values.argmax(dim=1).reshape(-1)
        return action

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                features_extractor=self.features_extractor,
                force_normalize_obs=self.force_normalize_obs,
                noisy_linear=self.noisy_linear,
                linear_kwargs=self.linear_kwargs,
                use_amp=self.use_amp,
            )
        )
        return data


class RainbowPolicy(BasePolicy):
    """Policy class with Q-Value Net and target net for DQN"""

    q_net: RainbowNetwork
    q_net_target: RainbowNetwork

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        net_arch: List[int] | None = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Dict[str, Any] | None = None,
        normalize_images: bool = True,
        noisy_linear: bool = True,
        linear_kwargs: Dict[str, Any] | None = None,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Dict[str, Any] | None = None,
        use_amp: bool = True,
        force_normalize_obs: bool = False,
    ) -> None:
        """
        Args:
            observation_space (spaces.Space): Observation space of the environment
            action_space (spaces.Discrete): Action space of the environment
            lr_schedule (Schedule): Learning rate schedule (could be constant)
            net_arch (List[int] | None, optional): Architecture of the multi layer perceptrons (MLPs) in the q-value networks.
                It represents the number of units per layer. The length of this list is the number of layers.
                If None, will be chosen based on features_extractor_class. Defaults to None.
            activation_fn (Type[nn.Module], optional): Activation function for the q-value networks. Defaults to nn.ReLU.
            features_extractor_class (Type[BaseFeaturesExtractor], optional): Module used to convert input observations to
                flattened feature tensor in q-value networks. Defaults to FlattenExtractor.
            features_extractor_kwargs (Dict[str, Any] | None, optional): Keyword arguments
                to pass to the features extractor. Defaults to None.
            normalize_images (bool, optional): Whether to normalize images or not, dividing by 255.0. Defaults to True.
            noisy_linear (bool, optional): Use noisy linear layers in the q-value networks. Defaults to True.
            linear_kwargs (Dict[str, Any] | None, optional): Additional arguments for the (noisy) linear
                layers in the q-value networks. Defaults to None.
            optimizer_class (Type[torch.optim.Optimizer], optional): Optimizer used for training the q-value networks. Defaults to torch.optim.Adam.
            optimizer_kwargs (Dict[str, Any] | None, optional): Additional keyword arguments,
                excluding the learning rate, to pass to the optimizer. Defaults to None.
            use_amp (bool, optional): Use Pytorch Automatic Mixed Precision, converting observations to float16. Defaults to True.
            force_normalize_obs (bool, optional): Due to frame stacking stable baselines may not be able to recognize image data,
                therefore skipping normalization and channel reordering. If this is set to True, the usual image preprocessing
                is always done, no matter the observation space. Defaults to False.
        """
        self.force_normalize_obs = force_normalize_obs
        if self.force_normalize_obs:
            # reorder image channel axis, as it is not done automatically
            height, width, channels = observation_space.shape
            observation_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(channels, height, width),
                dtype=observation_space.dtype,
            )

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

        self.use_amp = use_amp

        self.net_args = dict(
            observation_space=self.observation_space,
            action_space=self.action_space,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            normalize_images=self.normalize_images,
            noisy_linear=self.noisy_linear,
            linear_kwargs=self.linear_kwargs,
            use_amp=self.use_amp,
            force_normalize_obs=self.force_normalize_obs,
        )

        self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule) -> None:
        """Create the current and target network and the optimizer.
        Put the target network into evaluation mode.

        Args:
            lr_schedule (Schedule): Learning rate schedule.
                lr_schedule(1) is the initial learning rate
        """

        self.q_net = self.make_net()
        self.q_net_target = self.make_net()
        self.q_net_target.load_state_dict(self.q_net.state_dict())

        self.q_net_target.set_training_mode(False)

        optimizer_kwargs = self.optimizer_kwargs

        # Continual Backprop optimization needs additional arguments, so a distinction has to be made
        if self.optimizer_class == CBP:
            optimizer_kwargs = optimizer_kwargs | prepare_cbp_kwargs(
                self.q_net, self.net_args
            )

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(  # type: ignore[call-arg]
            self.q_net.parameters(),
            lr=lr_schedule(1),
            **optimizer_kwargs,
        )

    def make_net(self) -> RainbowNetwork:
        """Utility function for instantiating a single q-value network"""
        # Make sure we always have separate networks for features extractors etc
        net_args = self._update_features_extractor(
            self.net_args, features_extractor=None
        )
        return RainbowNetwork(**net_args).to(self.device)

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        raise NotImplementedError()

    def _predict(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Simply pass along any calls to the current q-value network"""
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
                use_amp=self.use_amp,
                force_normalize_obs=self.force_normalize_obs,
                noisy_linear=self.noisy_linear,
                linear_kwargs=self.linear_kwargs,
                normalize_images=self.normalize_images,
            )
        )
        return data

    def obs_to_tensor(self, observation: Any) -> Tuple[torch.Tensor, bool]:
        """Convert an input observation to a PyTorch tensor that can be fed to a model.
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        Extends the default stable baselines implementation with a custom conversion for LazyVecStackedObservations

        Args:
            observation (Any): observation straight from the environment

        Returns:
            Tuple[torch.Tensor, bool]: observations tensor and vectorized_env flag
        """
        if isinstance(observation, LazyVecStackedObservations):
            return (
                torch.from_numpy(observation.__array__()).to(self.device),
                True,
            )
        return super().obs_to_tensor(observation)


MlpPolicy = RainbowPolicy


class CnnPolicy(RainbowPolicy):
    """Policy class for DQN when using images as input.

    Simply sets force_normalize_obs=True and features_extractor_class=ImpalaCNNLarge"""

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        lr_schedule: Schedule,
        net_arch: List[int] | None = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = ImpalaCNNLarge,
        features_extractor_kwargs: Dict[str, Any] | None = None,
        normalize_images: bool = True,
        noisy_linear: bool = True,
        linear_kwargs: Dict[str, Any] | None = None,
        optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_kwargs: Dict[str, Any] | None = None,
        use_amp: bool = True,
        force_normalize_obs: bool = True,
    ) -> None:
        """
        Args:
            observation_space (spaces.Space): Observation space of the environment
            action_space (spaces.Discrete): Action space of the environment
            lr_schedule (Schedule): Learning rate schedule (could be constant)
            net_arch (List[int] | None, optional): Architecture of the multi layer perceptrons (MLPs) in the q-value networks.
                It represents the number of units per layer. The length of this list is the number of layers.
                If None, will be chosen based on features_extractor_class. Defaults to None.
            activation_fn (Type[nn.Module], optional): Activation function for the q-value networks. Defaults to nn.ReLU.
            features_extractor_class (Type[BaseFeaturesExtractor], optional): Module used to convert input observations to
                flattened feature tensor in q-value networks. Defaults to FlattenExtractor.
            features_extractor_kwargs (Dict[str, Any] | None, optional): Keyword arguments
                to pass to the features extractor. Defaults to None.
            normalize_images (bool, optional): Whether to normalize images or not, dividing by 255.0. Defaults to True.
            noisy_linear (bool, optional): Use noisy linear layers in the q-value networks. Defaults to True.
            linear_kwargs (Dict[str, Any] | None, optional): Additional arguments for the (noisy) linear
                layers in the q-value networks. Defaults to None.
            optimizer_class (Type[torch.optim.Optimizer], optional): Optimizer used for training the q-value networks. Defaults to torch.optim.Adam.
            optimizer_kwargs (Dict[str, Any] | None, optional): Additional keyword arguments,
                excluding the learning rate, to pass to the optimizer. Defaults to None.
            use_amp (bool, optional): Use Pytorch Automatic Mixed Precision, converting observations to float16. Defaults to True.
            force_normalize_obs (bool, optional): Due to frame stacking stable baselines may not be able to recognize image data,
                therefore skipping normalization and channel reordering. If this is set to True, the usual image preprocessing
                is always done, no matter the observation space. Defaults to False.
        """
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            noisy_linear=noisy_linear,
            linear_kwargs=linear_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            use_amp=use_amp,
            force_normalize_obs=force_normalize_obs,
        )

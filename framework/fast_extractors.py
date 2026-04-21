from __future__ import annotations

import gymnasium as gym
import torch as th
from gymnasium import spaces
from torch import nn

from stable_baselines3.common.preprocessing import is_image_space
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class FastCnnExtractor(BaseFeaturesExtractor):
    """
    Lighter CNN feature extractor for VizDoom image observations.

    This is meant as an optional speed/capacity tradeoff versus SB3 NatureCNN:
    fewer channels and a smaller feature vector reduce backward cost and VRAM.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 256,
        normalized_image: bool = False,
    ) -> None:
        if not isinstance(observation_space, spaces.Box):
            raise TypeError(f"FastCnnExtractor expects Box observation space, got {observation_space!r}")
        if not is_image_space(observation_space, check_channels=False, normalized_image=normalized_image):
            raise ValueError(f"FastCnnExtractor expects image observations, got {observation_space!r}")

        super().__init__(observation_space, features_dim)
        n_input_channels = int(observation_space.shape[0])

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with th.no_grad():
            sample = th.as_tensor(observation_space.sample()[None]).float()
            n_flatten = int(self.cnn(sample).shape[1])

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, int(features_dim)),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


class StateImageExtractor(BaseFeaturesExtractor):
    """
    CNN + MLP extractor for Dict observations:
      - image: stacked VizDoom frames, channels-first
      - state: normalized numeric combat state
    """

    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 384,
        image_features_dim: int = 256,
        state_features_dim: int = 128,
        normalized_image: bool = False,
    ) -> None:
        if not isinstance(observation_space, spaces.Dict):
            raise TypeError(f"StateImageExtractor expects Dict observation space, got {observation_space!r}")
        if "image" not in observation_space.spaces or "state" not in observation_space.spaces:
            raise ValueError("StateImageExtractor expects observation keys: 'image' and 'state'.")

        image_space = observation_space.spaces["image"]
        state_space = observation_space.spaces["state"]
        if not isinstance(image_space, spaces.Box):
            raise TypeError(f"'image' observation must be Box, got {image_space!r}")
        if not isinstance(state_space, spaces.Box):
            raise TypeError(f"'state' observation must be Box, got {state_space!r}")
        if not is_image_space(image_space, check_channels=False, normalized_image=normalized_image):
            raise ValueError(f"'image' observation must be image-like, got {image_space!r}")

        super().__init__(observation_space, int(features_dim))
        n_input_channels = int(image_space.shape[0])
        state_dim = int(state_space.shape[0])
        image_features_dim = int(image_features_dim)
        state_features_dim = int(state_features_dim)

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with th.no_grad():
            sample = th.as_tensor(image_space.sample()[None]).float()
            n_flatten = int(self.cnn(sample).shape[1])

        self.image_net = nn.Sequential(
            nn.Linear(n_flatten, image_features_dim),
            nn.ReLU(),
        )
        self.state_net = nn.Sequential(
            nn.Linear(state_dim, state_features_dim),
            nn.ReLU(),
            nn.Linear(state_features_dim, state_features_dim),
            nn.ReLU(),
        )
        self.combined = nn.Sequential(
            nn.Linear(image_features_dim + state_features_dim, int(features_dim)),
            nn.ReLU(),
        )

    def forward(self, observations: dict[str, th.Tensor]) -> th.Tensor:
        image_features = self.image_net(self.cnn(observations["image"]))
        state_features = self.state_net(observations["state"].float())
        return self.combined(th.cat((image_features, state_features), dim=1))


__all__ = ["FastCnnExtractor", "StateImageExtractor"]

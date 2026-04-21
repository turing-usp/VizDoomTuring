from __future__ import annotations

from collections.abc import Generator
from typing import Any

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.type_aliases import RolloutBufferSamples
from stable_baselines3.common.vec_env import VecNormalize


def _device_is_cuda(device: th.device | str) -> bool:
    if isinstance(device, th.device):
        return device.type == "cuda"
    return str(device).lower().startswith("cuda") or str(device).lower() == "auto"


def _numpy_dtype_to_torch(dtype: np.typing.DTypeLike) -> th.dtype:
    np_dtype = np.dtype(dtype).type
    mapping = {
        np.bool_: th.bool,
        np.uint8: th.uint8,
        np.int8: th.int8,
        np.int16: th.int16,
        np.int32: th.int32,
        np.int64: th.int64,
        np.float16: th.float16,
        np.float32: th.float32,
        np.float64: th.float64,
    }
    if np_dtype in mapping:
        return mapping[np_dtype]
    raise TypeError(f"Unsupported dtype for pinned storage: {np.dtype(dtype)!r}")


def _allocate_host_array(
    shape: tuple[int, ...],
    dtype: np.typing.DTypeLike,
    *,
    pinned: bool,
    storage_refs: list[th.Tensor],
) -> np.ndarray:
    if not pinned:
        return np.zeros(shape, dtype=dtype)

    tensor = th.empty(shape, dtype=_numpy_dtype_to_torch(dtype), pin_memory=True)
    tensor.zero_()
    storage_refs.append(tensor)
    return tensor.numpy()


class PinnedRolloutBuffer(RolloutBuffer):
    """
    RolloutBuffer drop-in replacement that keeps storage on host memory and,
    when CUDA is active, allocates page-locked pinned buffers to speed up
    minibatch transfers.

    The buffer never migrates the full rollout to VRAM.  Only sampled
    minibatches are converted to tensors and moved with non-blocking copies.
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: th.device | str = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
        *,
        pin_memory_if_cuda: bool = True,
    ):
        self._pin_memory_if_cuda = bool(pin_memory_if_cuda)
        self._use_pinned_storage = bool(
            self._pin_memory_if_cuda and th.cuda.is_available() and _device_is_cuda(device)
        )
        self._storage_refs: list[th.Tensor] = []
        super().__init__(
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            gae_lambda=gae_lambda,
            gamma=gamma,
            n_envs=n_envs,
        )

    def reset(self) -> None:
        self._storage_refs.clear()
        self.observations = _allocate_host_array(
            (self.buffer_size, self.n_envs, *self.obs_shape),
            self.observation_space.dtype,
            pinned=self._use_pinned_storage,
            storage_refs=self._storage_refs,
        )
        self.actions = _allocate_host_array(
            (self.buffer_size, self.n_envs, self.action_dim),
            self.action_space.dtype,
            pinned=self._use_pinned_storage,
            storage_refs=self._storage_refs,
        )
        self.rewards = _allocate_host_array(
            (self.buffer_size, self.n_envs),
            np.float32,
            pinned=self._use_pinned_storage,
            storage_refs=self._storage_refs,
        )
        self.returns = _allocate_host_array(
            (self.buffer_size, self.n_envs),
            np.float32,
            pinned=self._use_pinned_storage,
            storage_refs=self._storage_refs,
        )
        self.episode_starts = _allocate_host_array(
            (self.buffer_size, self.n_envs),
            np.float32,
            pinned=self._use_pinned_storage,
            storage_refs=self._storage_refs,
        )
        self.values = _allocate_host_array(
            (self.buffer_size, self.n_envs),
            np.float32,
            pinned=self._use_pinned_storage,
            storage_refs=self._storage_refs,
        )
        self.log_probs = _allocate_host_array(
            (self.buffer_size, self.n_envs),
            np.float32,
            pinned=self._use_pinned_storage,
            storage_refs=self._storage_refs,
        )
        self.advantages = _allocate_host_array(
            (self.buffer_size, self.n_envs),
            np.float32,
            pinned=self._use_pinned_storage,
            storage_refs=self._storage_refs,
        )
        self.generator_ready = False
        self.pos = 0
        self.full = False

    def compute_returns_and_advantage(self, last_values: th.Tensor, dones: np.ndarray) -> None:
        last_values_np = last_values.clone().cpu().numpy().flatten()
        last_gae_lam = 0.0

        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones.astype(np.float32)
                next_values = last_values_np
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]

            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam

        np.copyto(self.returns, self.advantages + self.values)

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
    ) -> None:
        if len(log_prob.shape) == 0:
            log_prob = log_prob.reshape(-1, 1)

        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))

        action = action.reshape((self.n_envs, self.action_dim))

        self.observations[self.pos] = np.array(obs)
        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.episode_starts[self.pos] = np.array(episode_start)
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(self, batch_size: int | None = None) -> Generator[RolloutBufferSamples, None, None]:
        assert self.full, "RolloutBuffer is not full"

        total_size = self.buffer_size * self.n_envs

        if not self.generator_ready:
            for tensor_name in (
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
            ):
                self.__dict__[tensor_name] = self.swap_and_flatten(self.__dict__[tensor_name])
            self.generator_ready = True

        if batch_size is None:
            batch_size = total_size

        if batch_size >= total_size:
            yield self._get_samples(slice(None))
            return

        indices = np.arange(total_size)
        np.random.shuffle(indices)

        start_idx = 0
        while start_idx < total_size:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(
        self,
        batch_inds: np.ndarray | slice,
        env: VecNormalize | None = None,
    ) -> RolloutBufferSamples:
        del env

        if isinstance(batch_inds, slice) and batch_inds.start is None and batch_inds.stop is None and batch_inds.step is None:
            observations = self.observations
            actions = self.actions.astype(np.float32, copy=False)
            values = self.values
            log_probs = self.log_probs
            advantages = self.advantages
            returns = self.returns
        else:
            observations = self.observations[batch_inds]
            actions = self.actions[batch_inds].astype(np.float32, copy=False)
            values = self.values[batch_inds]
            log_probs = self.log_probs[batch_inds]
            advantages = self.advantages[batch_inds]
            returns = self.returns[batch_inds]

        data = (
            observations,
            actions,
            values.reshape(-1),
            log_probs.reshape(-1),
            advantages.reshape(-1),
            returns.reshape(-1),
        )
        return RolloutBufferSamples(
            observations=self.to_torch(data[0], copy=False),
            actions=self.to_torch(data[1], copy=False),
            old_values=self.to_torch(data[2], copy=False),
            old_log_prob=self.to_torch(data[3], copy=False),
            advantages=self.to_torch(data[4], copy=False),
            returns=self.to_torch(data[5], copy=False),
        )

    def to_torch(self, array: Any, copy: bool = True) -> th.Tensor:
        if isinstance(array, th.Tensor):
            tensor = array.clone() if copy else array
        else:
            np_array = np.array(array, copy=True) if copy else np.asarray(array)
            tensor = th.as_tensor(np_array)

        if self.device.type == "cuda":
            if self._use_pinned_storage and not tensor.is_pinned():
                tensor = tensor.pin_memory()
            return tensor.to(self.device, non_blocking=True)
        return tensor.to(self.device)


__all__ = ["PinnedRolloutBuffer"]

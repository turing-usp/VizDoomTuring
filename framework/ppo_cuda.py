from __future__ import annotations

import contextlib
import time
import warnings
from typing import Any, Optional

import numpy as np
import torch
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3 import PPO
from stable_baselines3.common.utils import explained_variance


class CudaOptimizedPPO(PPO):
    """
    PPO subclass with CUDA-oriented training hooks.

    The default behavior stays identical to SB3 PPO. Optional AMP and
    torch.compile support are opt-in and disabled by default.
    """

    def __init__(
        self,
        *args: Any,
        use_amp: bool = False,
        amp_dtype: torch.dtype = torch.float16,
        amp_use_grad_scaler: Optional[bool] = None,
        compile_evaluate_actions: bool = False,
        compile_backend: str = "inductor",
        compile_mode: str = "default",
        compile_fullgraph: bool = False,
        compile_dynamic: Optional[bool] = True,
        zero_grad_set_to_none: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.use_amp = bool(use_amp)
        self.amp_dtype = amp_dtype
        self.amp_use_grad_scaler = amp_use_grad_scaler
        self.compile_evaluate_actions = bool(compile_evaluate_actions)
        self.compile_backend = compile_backend
        self.compile_mode = compile_mode
        self.compile_fullgraph = compile_fullgraph
        self.compile_dynamic = compile_dynamic
        self.zero_grad_set_to_none = zero_grad_set_to_none

        self._compiled_evaluate_actions = None
        self._amp_scaler = None
        self._compile_attempted = False

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_compiled_evaluate_actions"] = None
        state["_amp_scaler"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if "_compiled_evaluate_actions" not in self.__dict__:
            self._compiled_evaluate_actions = None
        if "_amp_scaler" not in self.__dict__:
            self._amp_scaler = None
        if "_compile_attempted" not in self.__dict__:
            self._compile_attempted = False

    def _excluded_save_params(self) -> list[str]:
        excluded = super()._excluded_save_params()
        return excluded + ["_compiled_evaluate_actions", "_amp_scaler"]

    def _should_use_amp(self) -> bool:
        return self.use_amp and getattr(self.device, "type", str(self.device)) == "cuda"

    def _get_autocast_context(self):
        if not self._should_use_amp():
            return contextlib.nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.amp_dtype)

    def _get_amp_scaler(self):
        if self._amp_scaler is not None:
            return self._amp_scaler

        if not self._should_use_amp():
            return None

        use_scaler = self.amp_use_grad_scaler
        if use_scaler is None:
            use_scaler = self.amp_dtype in {torch.float16, th.float16}

        if not use_scaler:
            return None

        scaler = None
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=True)  # type: ignore[attr-defined]
        except Exception:
            scaler = torch.cuda.amp.GradScaler(enabled=True)

        self._amp_scaler = scaler
        return scaler

    def _maybe_compile_evaluate_actions(self):
        if self._compiled_evaluate_actions is not None or self._compile_attempted or not self.compile_evaluate_actions:
            return self._compiled_evaluate_actions

        self._compile_attempted = True

        if not hasattr(torch, "compile"):
            warnings.warn("torch.compile is unavailable; continuing without compile.", stacklevel=2)
            return None

        compile_kwargs = {
            "backend": self.compile_backend,
            "mode": self.compile_mode,
            "fullgraph": self.compile_fullgraph,
        }
        if self.compile_dynamic is not None:
            compile_kwargs["dynamic"] = self.compile_dynamic

        try:
            self._compiled_evaluate_actions = torch.compile(
                self.policy.evaluate_actions,
                **compile_kwargs,
            )
        except Exception as exc:
            warnings.warn(
                f"torch.compile failed for evaluate_actions: {exc!r}. Continuing without compile.",
                stacklevel=2,
            )
            self._compiled_evaluate_actions = None

        return self._compiled_evaluate_actions

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        clip_range_vf = None
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        self._maybe_compile_evaluate_actions()
        evaluate_actions = self._compiled_evaluate_actions or self.policy.evaluate_actions
        scaler = self._get_amp_scaler()

        optimizer = self.policy.optimizer
        policy_parameters = self.policy.parameters
        discrete_action_space = isinstance(self.action_space, spaces.Discrete)
        cuda_enabled = getattr(self.device, "type", str(self.device)) == "cuda" and th.cuda.is_available()

        entropy_losses = []
        pg_losses = []
        value_losses = []
        clip_fractions = []
        approx_kl_divs = []
        continue_training = True
        loss = th.tensor(0.0, device=self.device)

        if cuda_enabled:
            th.cuda.synchronize()
        train_started = time.perf_counter()
        minibatch_count = 0

        for epoch in range(self.n_epochs):
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                minibatch_count += 1
                actions = rollout_data.actions
                if discrete_action_space:
                    actions = rollout_data.actions.long().flatten()

                with self._get_autocast_context():
                    values, log_prob, entropy = evaluate_actions(rollout_data.observations, actions)

                if self._should_use_amp():
                    values = values.float()
                    log_prob = log_prob.float()
                    if entropy is not None:
                        entropy = entropy.float()

                values = values.flatten()

                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
                pg_losses.append(policy_loss.detach())

                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float())
                clip_fractions.append(clip_fraction)

                if clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values,
                        -clip_range_vf,
                        clip_range_vf,
                    )

                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.detach())

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                entropy_losses.append(entropy_loss.detach())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio)
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and float(approx_kl_div.detach().cpu()) > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(
                            f"Early stopping at step {epoch} due to reaching max kl: {float(approx_kl_div.detach().cpu()):.2f}"
                        )
                    break

                if scaler is None:
                    optimizer.zero_grad(set_to_none=self.zero_grad_set_to_none)
                    loss.backward()
                    th.nn.utils.clip_grad_norm_(policy_parameters(), self.max_grad_norm)
                    optimizer.step()
                else:
                    optimizer.zero_grad(set_to_none=self.zero_grad_set_to_none)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    th.nn.utils.clip_grad_norm_(policy_parameters(), self.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()

                self._n_updates += 1

            if not continue_training:
                break

        if cuda_enabled:
            th.cuda.synchronize()
        train_duration_s = max(time.perf_counter() - train_started, 1e-9)
        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(),
            self.rollout_buffer.returns.flatten(),
        )

        def _mean_metric(values: list[th.Tensor]) -> float:
            if not values:
                return float("nan")
            return float(th.stack([v.float() for v in values]).mean().detach().cpu())

        self.logger.record("train/entropy_loss", _mean_metric(entropy_losses))
        self.logger.record("train/policy_gradient_loss", _mean_metric(pg_losses))
        self.logger.record("train/value_loss", _mean_metric(value_losses))
        self.logger.record("train/approx_kl", _mean_metric(approx_kl_divs))
        self.logger.record("train/clip_fraction", _mean_metric(clip_fractions))
        self.logger.record("train/loss", float(loss.detach().cpu()))
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/cuda_train_time_s", train_duration_s)
        self.logger.record(
            "train/cuda_minibatch_time_ms",
            1000.0 * train_duration_s / max(int(minibatch_count), 1),
        )
        self.logger.record("train/cuda_minibatches", int(minibatch_count), exclude="tensorboard")
        cuda_memory_allocated_mb = 0.0
        cuda_memory_reserved_mb = 0.0
        if cuda_enabled:
            cuda_memory_allocated_mb = th.cuda.memory_allocated(self.device) / (1024**2)
            cuda_memory_reserved_mb = th.cuda.memory_reserved(self.device) / (1024**2)
        self.logger.record("train/cuda_memory_allocated_mb", cuda_memory_allocated_mb)
        self.logger.record("train/cuda_memory_reserved_mb", cuda_memory_reserved_mb)

        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)

        if clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)


__all__ = ["CudaOptimizedPPO"]

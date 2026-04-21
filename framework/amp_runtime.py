from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, ContextManager, Optional

import torch


def _cuda_available() -> bool:
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _amp_available() -> bool:
    return _cuda_available() and hasattr(torch, "cuda") and hasattr(torch.cuda, "amp")


def _normalize_device_type(device_type: Optional[str]) -> str:
    if device_type:
        return str(device_type)
    return "cuda" if _cuda_available() else "cpu"


def _normalize_dtype(dtype: Optional[torch.dtype]) -> Optional[torch.dtype]:
    if dtype is not None:
        return dtype
    return torch.float16 if _cuda_available() else None


@contextmanager
def autocast(
    enabled: bool = True,
    *,
    device_type: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    cache_enabled: Optional[bool] = None,
) -> ContextManager[Any]:
    """
    Safe autocast helper.

    On CUDA, uses torch.autocast / torch.cuda.amp.autocast when available.
    On CPU or when AMP is disabled, behaves like a no-op context manager.
    """
    if not enabled or not _amp_available():
        with nullcontext():
            yield
        return

    dev = _normalize_device_type(device_type)
    amp_dtype = _normalize_dtype(dtype)

    autocast_factory = getattr(torch, "autocast", None)
    if callable(autocast_factory):
        kwargs = {"device_type": dev}
        if amp_dtype is not None:
            kwargs["dtype"] = amp_dtype
        if cache_enabled is not None:
            kwargs["cache_enabled"] = bool(cache_enabled)
        with autocast_factory(**kwargs):
            yield
        return

    amp_mod = torch.cuda.amp
    kwargs = {}
    if amp_dtype is not None:
        kwargs["dtype"] = amp_dtype
    if cache_enabled is not None:
        kwargs["cache_enabled"] = bool(cache_enabled)
    with amp_mod.autocast(**kwargs):
        yield


def _new_scaler(enabled: bool = True) -> "SafeGradScaler":
    return SafeGradScaler(enabled=enabled and _amp_available())


@dataclass(frozen=True)
class SafeGradScaler:
    """
    Small wrapper around torch's GradScaler with safe fallbacks.

    When AMP/CUDA is unavailable or disabled, all methods become no-ops,
    but the API stays the same so training code can stay linear.
    """

    enabled: bool = True
    init_scale: float = 2.0 ** 16
    growth_factor: float = 2.0
    backoff_factor: float = 0.5
    growth_interval: int = 2000
    hysteresis: int = 2

    def __post_init__(self) -> None:
        object.__setattr__(self, "_scaler", self._build_scaler())

    def _build_scaler(self) -> Any:
        if not self.enabled:
            return None

        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            try:
                return torch.amp.GradScaler(
                    "cuda",
                    init_scale=self.init_scale,
                    growth_factor=self.growth_factor,
                    backoff_factor=self.backoff_factor,
                    growth_interval=self.growth_interval,
                    hysteresis=self.hysteresis,
                )
            except TypeError:
                pass
            except Exception:
                return None

        if hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "GradScaler"):
            try:
                return torch.cuda.amp.GradScaler(
                    init_scale=self.init_scale,
                    growth_factor=self.growth_factor,
                    backoff_factor=self.backoff_factor,
                    growth_interval=self.growth_interval,
                )
            except Exception:
                return None

        return None

    @property
    def is_enabled(self) -> bool:
        return bool(self.enabled and self._scaler is not None)

    def scale(self, loss: Any) -> Any:
        if self._scaler is None:
            return loss
        return self._scaler.scale(loss)

    def backward(self, loss: Any) -> Any:
        scaled = self.scale(loss)
        if hasattr(scaled, "backward"):
            return scaled.backward()
        if hasattr(loss, "backward"):
            return loss.backward()
        raise TypeError("loss must expose a backward() method")

    def unscale_(self, optimizer: Any) -> Any:
        if self._scaler is None:
            return optimizer
        return self._scaler.unscale_(optimizer)

    def step(self, optimizer: Any, *args: Any, **kwargs: Any) -> Any:
        if self._scaler is None:
            return optimizer.step(*args, **kwargs)
        return self._scaler.step(optimizer, *args, **kwargs)

    def update(self, new_scale: Optional[float] = None) -> Any:
        if self._scaler is None:
            return None
        if new_scale is None:
            return self._scaler.update()
        return self._scaler.update(new_scale)

    def get_scale(self) -> float:
        if self._scaler is None:
            return 1.0
        try:
            return float(self._scaler.get_scale())
        except Exception:
            return 1.0

    def state_dict(self) -> dict[str, Any]:
        if self._scaler is None:
            return {"enabled": False, "scale": 1.0}
        try:
            state = dict(self._scaler.state_dict())
        except Exception:
            state = {}
        state["enabled"] = True
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if self._scaler is None:
            return
        try:
            self._scaler.load_state_dict(state_dict)
        except Exception:
            return


def create_scaler(enabled: bool = True) -> SafeGradScaler:
    return _new_scaler(enabled=enabled)


def scale_loss(scaler: Optional[SafeGradScaler], loss: Any) -> Any:
    if scaler is None:
        return loss
    return scaler.scale(loss)


def backward(scaler: Optional[SafeGradScaler], loss: Any) -> Any:
    if scaler is None:
        if hasattr(loss, "backward"):
            return loss.backward()
        raise TypeError("loss must expose a backward() method")
    return scaler.backward(loss)


def unscale_(scaler: Optional[SafeGradScaler], optimizer: Any) -> Any:
    if scaler is None:
        return optimizer
    return scaler.unscale_(optimizer)


def step(scaler: Optional[SafeGradScaler], optimizer: Any, *args: Any, **kwargs: Any) -> Any:
    if scaler is None:
        return optimizer.step(*args, **kwargs)
    return scaler.step(optimizer, *args, **kwargs)


def update(scaler: Optional[SafeGradScaler], new_scale: Optional[float] = None) -> Any:
    if scaler is None:
        return None
    return scaler.update(new_scale)


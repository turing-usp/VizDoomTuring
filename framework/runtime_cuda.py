from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

try:
    import torch
except Exception:  # pragma: no cover - torch may be unavailable in some envs
    torch = None  # type: ignore[assignment]


@dataclass(frozen=True)
class CUDARuntimeStatus:
    """Readable CUDA runtime summary for logging."""

    torch_available: bool
    cuda_available: bool
    cudnn_benchmark: bool
    tf32_enabled: bool
    matmul_precision: str

    def __str__(self) -> str:
        return (
            "CUDA runtime "
            f"(torch={self.torch_available}, cuda={self.cuda_available}, "
            f"cudnn.benchmark={self.cudnn_benchmark}, tf32={self.tf32_enabled}, "
            f"matmul_precision='{self.matmul_precision}')"
        )


def _read_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _normalize_precision(value: Optional[str], default: str) -> str:
    precision = (value or default).strip().lower()
    if precision not in {"highest", "high", "medium"}:
        return default
    return precision


def configure_cuda_runtime(
    *,
    benchmark: Optional[bool] = None,
    tf32: Optional[bool] = None,
    matmul_precision: Optional[str] = None,
) -> CUDARuntimeStatus:
    """
    Configure CUDA runtime options centrally in PyTorch.

    Conservative defaults:
    - cuDNN benchmark: desligado
    - TF32: desligado
    - torch matmul precision: `highest`

    Options can be overridden by arguments or environment variables:
    - `VIZDOOM_CUDNN_BENCHMARK`
    - `VIZDOOM_CUDA_TF32`
    - `VIZDOOM_TORCH_MATMUL_PRECISION`
    """
    benchmark_value = _read_bool_env("VIZDOOM_CUDNN_BENCHMARK", False) if benchmark is None else benchmark
    tf32_value = _read_bool_env("VIZDOOM_CUDA_TF32", False) if tf32 is None else tf32
    precision_value = _normalize_precision(
        os.getenv("VIZDOOM_TORCH_MATMUL_PRECISION") if matmul_precision is None else matmul_precision,
        "highest",
    )

    torch_available = torch is not None
    cuda_available = bool(torch_available and torch.cuda.is_available())

    if torch_available:
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = bool(benchmark_value)
            if hasattr(torch.backends.cudnn, "allow_tf32"):
                torch.backends.cudnn.allow_tf32 = bool(tf32_value)

        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = bool(tf32_value)

        set_precision = getattr(torch, "set_float32_matmul_precision", None)
        if callable(set_precision):
            set_precision(precision_value)

    return CUDARuntimeStatus(
        torch_available=torch_available,
        cuda_available=cuda_available,
        cudnn_benchmark=bool(benchmark_value),
        tf32_enabled=bool(tf32_value),
        matmul_precision=precision_value,
    )


def get_cuda_runtime_summary(
    *,
    benchmark: Optional[bool] = None,
    tf32: Optional[bool] = None,
    matmul_precision: Optional[str] = None,
) -> str:
    """Logging shortcut: returns the configuration as readable text."""
    return str(
        configure_cuda_runtime(
            benchmark=benchmark,
            tf32=tf32,
            matmul_precision=matmul_precision,
        )
    )

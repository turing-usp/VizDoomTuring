from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Tuple


@dataclass(frozen=True, slots=True)
class Span:
    """Half-open interval [start, stop) used to describe a pure staging slice."""

    start: int
    stop: int

    @property
    def size(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True, slots=True)
class LearnerStagingPlan:
    """
    Immutable description of how a learner rollout can be staged.

    The plan is intentionally semantic-preserving:
    - no samples are dropped
    - no samples are duplicated
    - order is preserved within each partition
    - the last partition may be smaller than the others
    """

    total_items: int
    stage_size: int
    minibatch_size: int
    stages: Tuple[Span, ...]
    minibatches: Tuple[Span, ...]

    @property
    def num_stages(self) -> int:
        return len(self.stages)

    @property
    def num_minibatches(self) -> int:
        return len(self.minibatches)


def _require_positive_int(value: int, name: str) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value}.")
    return value


def split_range(total_items: int, chunk_size: int) -> Tuple[Span, ...]:
    """
    Split [0, total_items) into contiguous half-open spans.

    This is a pure helper that preserves the original order and keeps the
    final span smaller when the total is not divisible by chunk_size.
    """

    total_items = _require_positive_int(total_items, "total_items")
    chunk_size = _require_positive_int(chunk_size, "chunk_size")

    spans = []
    start = 0
    while start < total_items:
        stop = min(start + chunk_size, total_items)
        spans.append(Span(start=start, stop=stop))
        start = stop
    return tuple(spans)


def minibatch_sizes(total_items: int, minibatch_size: int) -> Tuple[int, ...]:
    """Return the exact minibatch sizes that cover total_items."""

    return tuple(span.size for span in split_range(total_items, minibatch_size))


def minibatch_spans(total_items: int, minibatch_size: int) -> Tuple[Span, ...]:
    """Return the contiguous minibatch spans that cover the full range."""

    return split_range(total_items, minibatch_size)


def stage_spans(total_items: int, stage_size: int) -> Tuple[Span, ...]:
    """Alias for split_range that reads better at the call site."""

    return split_range(total_items, stage_size)


def build_learner_staging_plan(
    total_items: int,
    *,
    stage_size: int,
    minibatch_size: int,
) -> LearnerStagingPlan:
    """
    Build a semantic-preserving learner staging plan.

    The plan only partitions the data; it does not decide any shuffle order.
    That keeps it compatible with PPO-style minibatch iteration where the
    caller may apply its own permutation before slicing.
    """

    total_items = _require_positive_int(total_items, "total_items")
    stage_size = _require_positive_int(stage_size, "stage_size")
    minibatch_size = _require_positive_int(minibatch_size, "minibatch_size")

    stages = stage_spans(total_items, stage_size)
    minibatches = minibatch_spans(total_items, minibatch_size)
    return LearnerStagingPlan(
        total_items=total_items,
        stage_size=stage_size,
        minibatch_size=minibatch_size,
        stages=stages,
        minibatches=minibatches,
    )


def infer_ppo_rollout_items(n_steps: int, n_envs: int) -> int:
    """
    Infer the flattened PPO rollout size as n_steps * n_envs.

    This mirrors SB3's rollout buffer shape without introducing any policy
    changes or sampling semantics.
    """

    n_steps = _require_positive_int(n_steps, "n_steps")
    n_envs = _require_positive_int(n_envs, "n_envs")
    return n_steps * n_envs


def infer_minibatch_size(
    total_items: int,
    *,
    n_minibatches: int | None = None,
    minibatch_size: int | None = None,
) -> int:
    """
    Infer a minibatch size from either a target count or an explicit size.

    The helper is intentionally conservative:
    - if minibatch_size is provided, it is returned unchanged after validation
    - if n_minibatches is provided, the size is the ceiling division
    - if neither is provided, the full batch is used as one minibatch
    """

    total_items = _require_positive_int(total_items, "total_items")

    if minibatch_size is not None:
        return _require_positive_int(minibatch_size, "minibatch_size")

    if n_minibatches is not None:
        n_minibatches = _require_positive_int(n_minibatches, "n_minibatches")
        return ceil(total_items / n_minibatches)

    return total_items


def build_ppo_staging_plan(
    n_steps: int,
    n_envs: int,
    *,
    stage_size: int | None = None,
    n_minibatches: int | None = None,
    minibatch_size: int | None = None,
) -> LearnerStagingPlan:
    """
    Convenience helper for PPO-style rollout staging.

    The resulting plan is a pure description only. It keeps the flattened
    rollout size intact and slices it into contiguous stages/minibatches.
    """

    total_items = infer_ppo_rollout_items(n_steps, n_envs)
    inferred_minibatch = infer_minibatch_size(
        total_items,
        n_minibatches=n_minibatches,
        minibatch_size=minibatch_size,
    )
    inferred_stage_size = _require_positive_int(
        stage_size if stage_size is not None else total_items,
        "stage_size",
    )
    return build_learner_staging_plan(
        total_items,
        stage_size=inferred_stage_size,
        minibatch_size=inferred_minibatch,
    )


__all__ = [
    "LearnerStagingPlan",
    "Span",
    "build_learner_staging_plan",
    "build_ppo_staging_plan",
    "infer_minibatch_size",
    "infer_ppo_rollout_items",
    "minibatch_sizes",
    "minibatch_spans",
    "split_range",
    "stage_spans",
]

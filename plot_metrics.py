#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gera gráficos dos CSVs de treino do VizDoom.")
    parser.add_argument(
        "--metrics",
        default=r"agents/deathmatch/stern_test_ppo_metrics.csv",
        help="CSV de métricas de RL.",
    )
    parser.add_argument(
        "--perf",
        default=r"agents/deathmatch/stern_test_ppo_perf.csv",
        help="CSV de métricas de estabilidade/performance.",
    )
    parser.add_argument(
        "--outdir",
        default=r"agents/deathmatch/plots",
        help="Diretório de saída dos PNGs.",
    )
    return parser.parse_args()


def read_csv(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def to_float_series(rows: List[Dict[str, str]], x_key: str, y_key: str) -> Tuple[List[float], List[float]]:
    xs: List[float] = []
    ys: List[float] = []
    for row in rows:
        xv = row.get(x_key, "")
        yv = row.get(y_key, "")
        if xv in ("", None) or yv in ("", None):
            continue
        try:
            xs.append(float(xv))
            ys.append(float(yv))
        except ValueError:
            continue
    return xs, ys


def save_plot(rows: List[Dict[str, str]], x_key: str, y_key: str, outpath: str, title: str) -> None:
    xs, ys = to_float_series(rows, x_key, y_key)
    if not xs or not ys:
        print(f"[PLOT][WARN] Sem dados para {y_key}")
        return

    plt.figure(figsize=(10, 5))
    plt.plot(xs, ys, linewidth=1.6)
    plt.title(title)
    plt.xlabel(x_key)
    plt.ylabel(y_key)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=140)
    plt.close()
    print(f"[PLOT] {outpath}")


def main() -> None:
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    metrics_rows = read_csv(args.metrics)
    perf_rows = read_csv(args.perf)

    metric_plots = [
        ("num_timesteps", "reward_mean_window", "reward_mean_window.png", "Reward Mean Window"),
        ("num_timesteps", "train_loss", "train_loss.png", "Train Loss"),
        ("num_timesteps", "train_value_loss", "train_value_loss.png", "Value Loss"),
        ("num_timesteps", "train_policy_gradient_loss", "train_policy_gradient_loss.png", "Policy Gradient Loss"),
        ("num_timesteps", "train_entropy_loss", "train_entropy_loss.png", "Entropy Loss"),
        ("num_timesteps", "train_approx_kl", "train_approx_kl.png", "Approx KL"),
        ("num_timesteps", "train_clip_fraction", "train_clip_fraction.png", "Clip Fraction"),
        ("num_timesteps", "train_explained_variance", "train_explained_variance.png", "Explained Variance"),
        ("num_timesteps", "rollout_ep_rew_mean", "rollout_ep_rew_mean.png", "Episode Reward Mean"),
        ("num_timesteps", "rollout_ep_len_mean", "rollout_ep_len_mean.png", "Episode Length Mean"),
        ("num_timesteps", "fps", "training_fps.png", "Training FPS"),
    ]

    perf_plots = [
        ("window_s", "vec_steps_per_s", "vec_steps_per_s.png", "Vec Steps Per Second"),
        ("window_s", "env_steps_per_s", "env_steps_per_s.png", "Env Steps Per Second"),
        ("window_s", "step_wait_ms_avg", "step_wait_ms_avg.png", "Step Wait Avg (ms)"),
        ("window_s", "step_wait_ms_max", "step_wait_ms_max.png", "Step Wait Max (ms)"),
        ("window_s", "batch_ms_avg", "batch_ms_avg.png", "Batch Avg (ms)"),
        ("window_s", "batch_ms_max", "batch_ms_max.png", "Batch Max (ms)"),
        ("window_s", "reply_span_ms_avg", "reply_span_ms_avg.png", "Reply Span Avg (ms)"),
        ("window_s", "reply_span_ms_max", "reply_span_ms_max.png", "Reply Span Max (ms)"),
    ]

    for x_key, y_key, filename, title in metric_plots:
        save_plot(metrics_rows, x_key, y_key, os.path.join(args.outdir, filename), title)

    for x_key, y_key, filename, title in perf_plots:
        save_plot(perf_rows, x_key, y_key, os.path.join(args.outdir, filename), title)

    print(f"[PLOT] Concluído. Saída em: {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()

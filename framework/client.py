#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import multiprocessing as mp
from typing import Any

import yaml

from .config import (
    DMConfig,
    AgentConfig,
    EngineRewardConfig,
    ShapingConfig,
    PolicyConfig,
    RewardConfig,
    RenderSettingsConfig,
)
from .env import DoomDMEnv
from .train import train_or_play


def _agentconfig_supports(field_name: str) -> bool:
    """Compat: só passa kwargs que existirem no AgentConfig atual."""
    try:
        return field_name in getattr(AgentConfig, "__dataclass_fields__", {})
    except Exception:
        return False


def load_agent_cfg(yaml_path: str) -> AgentConfig:
    """
    Loader completo de AgentConfig a partir de um YAML.

    Espera layout:

    render_settings: {...}

    reward:
      engine: {...}
      shaping: {...}

    policy: {...}

    Campos extras no YAML (ex.: weapon/lock_weapon) são ignorados se o AgentConfig
    atual não suportar.
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f) or {}

    # Render Settings
    rs = RenderSettingsConfig(**y.get("render_settings", {}))

    # Reward (engine + shaping)
    reward_block = y.get("reward", {})
    er = EngineRewardConfig(**reward_block.get("engine", {}))
    sh = ShapingConfig(**reward_block.get("shaping", {}))
    reward_cfg = RewardConfig(engine=er, shaping=sh)

    pol = PolicyConfig(**y.get("policy", {}))

    agent_kwargs: dict[str, Any] = dict(
        name=y.get("name", "Client"),
        colorset=y.get("colorset", 3),
        render_settings=rs,
        reward=reward_cfg,
        policy=pol,
        model_dir=y.get("model_dir", "models"),
        model_name=y.get("model_name", "agent.zip"),
        train=bool(y.get("train", False)),
        train_steps=int(y.get("train_steps", 300_000)),
        stack_frames=int(y.get("stack_frames", y.get("stack", 4))),
    )

    # Compat: só injeta se existir no dataclass atual
    if _agentconfig_supports("weapon"):
        agent_kwargs["weapon"] = str(y.get("weapon", "RocketLauncher"))
    if _agentconfig_supports("lock_weapon"):
        agent_kwargs["lock_weapon"] = bool(y.get("lock_weapon", True))

    return AgentConfig(**agent_kwargs)


def _dmconfig_supports(field_name: str) -> bool:
    """Compat: só passa kwargs que existirem no DMConfig atual."""
    try:
        return field_name in getattr(DMConfig, "__dataclass_fields__", {})
    except Exception:
        return False


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, help="YAML do agente")
    ap.add_argument("--port", type=int, default=5029)
    ap.add_argument("--ip", default="127.0.0.1")

    ap.add_argument(
        "--host",
        action="store_true",
        help="Se definido, este cliente agirá como o host da partida.",
    )
    ap.add_argument(
        "--players",
        type=int,
        default=2,
        help="Número total de jogadores que o host deve esperar.",
    )

    ap.add_argument("--render", action="store_true")
    ap.add_argument("--timelimit", type=float, default=3.0)
    ap.add_argument("--stack", type=int, default=4)

    # map/wad selection
    ap.add_argument("--map", default="map01", help="Nome do mapa (ex.: MAP01, map01).")
    ap.add_argument(
        "--wad",
        default=None,
        help="WAD/PK3 extra. Pode ser caminho completo/relativo ou nome em framework/maps/.",
    )

    args = ap.parse_args()

    agent = load_agent_cfg(args.cfg)

    dm_kwargs: dict[str, Any] = dict(
        total_players=args.players,
        port=args.port,
        join_ip=args.ip,
        map_name=args.map,
        timelimit_minutes=args.timelimit,
        render=args.render,
        stack_frames=args.stack,
    )
    if args.wad and _dmconfig_supports("wad"):
        dm_kwargs["wad"] = args.wad

    dm = DMConfig(**dm_kwargs)

    os.makedirs(agent.model_dir, exist_ok=True)
    save_path = os.path.join(agent.model_dir, agent.model_name)

    def _env():
        return DoomDMEnv(name=agent.name, is_host=args.host, dm=dm, agent=agent)

    mp.set_start_method("spawn", force=True)
    train_or_play(_env, dm.stack_frames, agent, save_path)

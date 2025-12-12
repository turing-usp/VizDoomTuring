#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from multiprocessing.connection import Client
from typing import Any, Dict

import yaml

from .config import (
    DMConfig,
    AgentConfig,
    EngineRewardConfig,
    ShapingConfig,
    PolicyConfig,
    RenderSettingsConfig,
    RewardConfig,
)
from .env import DoomDMEnv


def load_agent_cfg_light(yaml_path: str) -> AgentConfig:
    """
    Lê o YAML e carrega as configurações, incluindo RenderSettings e RewardConfig.
    Espera layout:

    reward:
      engine: { ... }
      shaping: { ... }
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)

    # Render Settings
    rs = RenderSettingsConfig(**y.get("render_settings", {}))

    # Reward (engine + shaping)
    reward_block = y.get("reward", {})
    er = EngineRewardConfig(**reward_block.get("engine", {}))
    sh = ShapingConfig(**reward_block.get("shaping", {}))
    reward_cfg = RewardConfig(engine=er, shaping=sh)

    pol = PolicyConfig(**y.get("policy", {}))

    return AgentConfig(
        name=y.get("name", "Client"),
        colorset=y.get("colorset", 3),
        render_settings=rs,
        reward=reward_cfg,
        policy=pol,
        model_dir=y.get("model_dir", "models"),
        model_name=y.get("model_name", "agent.zip"),
        train=bool(y.get("train", False)),
        train_steps=int(y.get("train_steps", 300_000)),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ator remoto VizDoom DM")
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--players", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--join-ip", default="127.0.0.1")
    parser.add_argument("--timelimit", type=float, default=0.0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--is-host", action="store_true")
    parser.add_argument("--trainer-host", default="127.0.0.1")
    parser.add_argument("--trainer-port", type=int, required=True)
    parser.add_argument("--auth-key", default="vizdoom_dm")
    return parser.parse_args()


def make_env(args: argparse.Namespace) -> DoomDMEnv:
    agent_cfg = load_agent_cfg_light(args.cfg)

    print(
        f"[ACTOR] Agente: {agent_cfg.name} | "
        f"Resolução YAML: {agent_cfg.render_settings.resolution}",
        flush=True,
    )

    dm_cfg = DMConfig(
        total_players=args.players,
        port=args.port,
        join_ip=args.join_ip,
        timelimit_minutes=float(args.timelimit),
        render=args.render,
    )

    print(f"[ACTOR] Criando DoomDMEnv (is_host={args.is_host})...", flush=True)
    env = DoomDMEnv(
        name=agent_cfg.name,
        is_host=args.is_host,
        dm=dm_cfg,
        agent=agent_cfg,
    )
    print("[ACTOR] DoomDMEnv criado.", flush=True)
    return env


def handle_reset(env: DoomDMEnv) -> Dict[str, Any]:
    print(f"[ACTOR] reset(): chamado (is_host={env.is_host})", flush=True)
    try:
        result = env.reset()
        obs, info = result if isinstance(result, tuple) else (result, {})
        return {"obs": obs, "info": info}
    except Exception as e:
        print(f"[ACTOR][ERROR] reset: {e!r}", flush=True)
        return {"error": str(e)}


def handle_step(env: DoomDMEnv, action: int) -> Dict[str, Any]:
    try:
        obs, reward, term, trunc, info = env.step(int(action))
        return {
            "obs": obs,
            "reward": float(reward),
            "done": bool(term or trunc),
            "info": info,
        }
    except Exception as e:
        print(f"[ACTOR][ERROR] step: {e!r}", flush=True)
        raise


def actor_loop(env: DoomDMEnv, conn: Client) -> None:
    print("[ACTOR] Loop iniciado.", flush=True)
    step_count = 0
    try:
        while True:
            try:
                msg = conn.recv()
            except EOFError:
                break

            if msg is None:
                break

            cmd = msg["cmd"]

            if cmd == "get_spaces":
                conn.send(
                    {
                        "obs_space": env.observation_space,
                        "action_space": env.action_space,
                    }
                )

            elif cmd == "reset":
                result = handle_reset(env)
                conn.send(result)

            elif cmd == "step":
                step_count += 1
                if step_count % 10000 == 0:
                    print(f"[ACTOR] Steps: {step_count}", flush=True)
                result = handle_step(env, msg["action"])
                conn.send(result)

            elif cmd == "close":
                try:
                    env.close()
                except Exception:
                    pass
                try:
                    conn.send({"ok": True})
                except Exception:
                    pass
                break
    finally:
        try:
            env.close()
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass
        print("[ACTOR] Encerrado.", flush=True)


def main() -> None:
    args = parse_args()
    address = (args.trainer_host, args.trainer_port)
    print(f"[ACTOR] Conectando em {address}...", flush=True)

    for _ in range(30):
        try:
            conn = Client(address, authkey=args.auth_key.encode("utf-8"))
            break
        except ConnectionRefusedError:
            time.sleep(1.0)
    else:
        raise RuntimeError("Falha ao conectar no Treinador.")

    print("[ACTOR] Conectado (TCP). Carregando VizDoom...", flush=True)
    try:
        env = make_env(args)
    except Exception as e:
        try:
            conn.send({"error": str(e)})
        except Exception:
            pass
        raise

    actor_loop(env, conn)


if __name__ == "__main__":
    main()

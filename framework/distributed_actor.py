#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
import traceback
from multiprocessing import shared_memory
from multiprocessing.connection import Client, Connection
from typing import Any, Dict, Optional, Tuple

import numpy as np
import yaml

from .config import (
    DMConfig,
    AgentConfig,
    EngineRewardConfig,
    EnemyInViewConfig,
    ShapingConfig,
    PolicyConfig,
    RenderSettingsConfig,
    RewardConfig,
    WallStuckConfig,
)
from .env import DoomDMEnv


# ----------------------------------------------------------------------
# Compat helpers
# ----------------------------------------------------------------------
def _agentconfig_supports(field_name: str) -> bool:
    """Compat: só passa kwargs que existirem no AgentConfig atual."""
    try:
        return field_name in getattr(AgentConfig, "__dataclass_fields__", {})
    except Exception:
        return False


def load_agent_cfg_light(yaml_path: str) -> AgentConfig:
    """
    Lê o YAML e carrega configurações (RenderSettings, RewardConfig, PolicyConfig).

    Campos extras no YAML são ignorados se não existirem no AgentConfig atual.
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f) or {}

    rs = RenderSettingsConfig(**y.get("render_settings", {}))

    reward_block = y.get("reward", {})
    er = EngineRewardConfig(**reward_block.get("engine", {}))
    shaping_raw = dict(reward_block.get("shaping", {}))

    wall_stuck_raw = dict(reward_block.get("wall_stuck", {}))
    if "wall_stuck_min_move" in shaping_raw and "min_move" not in wall_stuck_raw:
        wall_stuck_raw["min_move"] = shaping_raw.pop("wall_stuck_min_move")
    if "wall_stuck_max_turn_deg" in shaping_raw and "max_turn_deg" not in wall_stuck_raw:
        wall_stuck_raw["max_turn_deg"] = shaping_raw.pop("wall_stuck_max_turn_deg")
    if "wall_stuck_persist_steps" in shaping_raw and "persist_steps" not in wall_stuck_raw:
        wall_stuck_raw["persist_steps"] = shaping_raw.pop("wall_stuck_persist_steps")

    enemy_in_view_raw = dict(reward_block.get("enemy_in_view", {}))
    if "enemy_in_view_check_every" in shaping_raw and "check_every" not in enemy_in_view_raw:
        enemy_in_view_raw["check_every"] = shaping_raw.pop("enemy_in_view_check_every")
    if "enemy_in_view_cooldown_steps" in shaping_raw and "cooldown_steps" not in enemy_in_view_raw:
        enemy_in_view_raw["cooldown_steps"] = shaping_raw.pop("enemy_in_view_cooldown_steps")
    if "enemy_in_view_min_area_ratio" in shaping_raw and "min_area_ratio" not in enemy_in_view_raw:
        enemy_in_view_raw["min_area_ratio"] = shaping_raw.pop("enemy_in_view_min_area_ratio")

    sh = ShapingConfig(**shaping_raw)
    reward_cfg = RewardConfig(
        engine=er,
        shaping=sh,
        wall_stuck=WallStuckConfig(**wall_stuck_raw),
        enemy_in_view=EnemyInViewConfig(**enemy_in_view_raw),
    )

    pol = PolicyConfig(**y.get("policy", {}))

    kwargs: Dict[str, Any] = dict(
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

    # Campos opcionais (compatível se você remover isso do AgentConfig)
    if _agentconfig_supports("weapon") and "weapon" in y:
        kwargs["weapon"] = str(y.get("weapon", ""))
    if _agentconfig_supports("lock_weapon") and "lock_weapon" in y:
        kwargs["lock_weapon"] = bool(y.get("lock_weapon", False))

    return AgentConfig(**kwargs)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ator remoto VizDoom DM")
    parser.add_argument("--cfg", required=True)

    # Topologia / rede do jogo
    parser.add_argument("--players", type=int, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--join-ip", default="127.0.0.1")

    # Map/WAD switching
    parser.add_argument("--map", default="map01", help="Nome do mapa (ex.: MAP01, map01).")
    parser.add_argument("--scenario", default=None, help="Cenário base .cfg/.wad/.pk3.")
    parser.add_argument(
        "--wad",
        default=None,
        help="WAD/PK3 extra (nome em framework/maps/ ou caminho completo).",
    )
    parser.add_argument("--frame-skip", type=int, default=8)
    parser.add_argument("--ticrate", type=int, default=30)

    # Match settings
    parser.add_argument("--timelimit", type=float, default=0.0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--render-agent-view", action="store_true")
    parser.add_argument("--is-host", action="store_true")

    # IPC (trainer<->actor)
    parser.add_argument("--trainer-host", default="127.0.0.1")
    parser.add_argument("--trainer-port", type=int, required=True)
    parser.add_argument("--auth-key", default="vizdoom_dm")

    return parser.parse_args()


# ----------------------------------------------------------------------
# Env creation
# ----------------------------------------------------------------------
def make_env(args: argparse.Namespace) -> DoomDMEnv:
    agent_cfg = load_agent_cfg_light(args.cfg)

    print(
        f"[ACTOR] Agente: {agent_cfg.name} | "
        f"Resolução YAML: {agent_cfg.render_settings.resolution} | "
        f"Scenario: {args.scenario} | Map: {args.map} | WAD: {args.wad} | "
        f"frame_skip: {args.frame_skip} | ticrate: {args.ticrate} | "
        f"Render: {bool(args.render)} | RenderAgentView: {bool(args.render_agent_view)} | "
        f"Host: {bool(args.is_host)}",
        flush=True,
    )

    dm_cfg = DMConfig(
        total_players=args.players,
        port=args.port,
        join_ip=args.join_ip,
        scenario=args.scenario,
        map_name=str(args.map),
        wad=args.wad,
        timelimit_minutes=float(args.timelimit),
        render=bool(args.render),
        render_agent_view=bool(args.render_agent_view),
        frame_skip=int(args.frame_skip),
        ticrate=int(args.ticrate),
    )

    print(f"[ACTOR] Criando DoomDMEnv (is_host={args.is_host})...", flush=True)
    env = DoomDMEnv(
        name=agent_cfg.name,
        is_host=bool(args.is_host),
        dm=dm_cfg,
        agent=agent_cfg,
    )
    print("[ACTOR] DoomDMEnv criado.", flush=True)
    return env


# ----------------------------------------------------------------------
# Step/reset handlers (always return dict; never raise)
# ----------------------------------------------------------------------
def handle_reset(env: DoomDMEnv) -> Dict[str, Any]:
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
        print(f"[ACTOR][ERROR] step(action={action}): {e!r}", flush=True)
        return {"error": str(e)}


# ----------------------------------------------------------------------
# Shared-memory writer for observations (optional)
# ----------------------------------------------------------------------
class ShmObsWriter:
    """
    Writer para observações via multiprocessing.shared_memory.

    Trainer cria o SHM e manda:
      cmd=set_shm_obs, shm_name=<nome>, shape=[...], dtype="uint8"
    (Compat: aceita também "name" no lugar de "shm_name".)

    Quando habilitado, o ator:
      - escreve obs no buffer
      - retorna via IPC só reward/done/info + obs_seq
    """

    def __init__(self) -> None:
        self._shm: Optional[shared_memory.SharedMemory] = None
        self._view: Optional[np.ndarray] = None
        self._obs_seq: int = 0

    @property
    def enabled(self) -> bool:
        return self._view is not None

    @property
    def obs_seq(self) -> int:
        return self._obs_seq

    def attach(self, shm_name: str, shape: Tuple[int, ...], dtype: np.dtype) -> None:
        self.close()
        shm = shared_memory.SharedMemory(name=shm_name, create=False)
        view = np.ndarray(shape, dtype=dtype, buffer=shm.buf)

        if view.nbytes > shm.size:
            shm.close()
            raise ValueError(f"SHM pequeno: view.nbytes={view.nbytes} > shm.size={shm.size}")

        self._shm = shm
        self._view = view
        self._obs_seq = 0

    def _write_compatible(self, obs: Any) -> None:
        """
        Escreve sem copiar quando possível, tolerando casos comuns de shape:
          - obs (H,W,1) -> shm (H,W)
          - obs (H,W,1) -> shm (1,H,W)
        """
        assert self._view is not None

        arr = np.asarray(obs)

        # Exato
        if arr.shape == self._view.shape:
            if arr.dtype != self._view.dtype:
                arr = arr.astype(self._view.dtype, copy=False)
            self._view[...] = arr
            return

        # (H,W,1) -> (H,W)
        if arr.ndim == 3 and arr.shape[-1] == 1 and self._view.ndim == 2:
            h, w, _ = arr.shape
            if self._view.shape == (h, w):
                src = arr[..., 0]
                if src.dtype != self._view.dtype:
                    src = src.astype(self._view.dtype, copy=False)
                self._view[...] = src
                return

        # (H,W,1) -> (1,H,W)
        if arr.ndim == 3 and arr.shape[-1] == 1 and self._view.ndim == 3 and self._view.shape[0] == 1:
            h, w, _ = arr.shape
            if self._view.shape == (1, h, w):
                src = np.transpose(arr, (2, 0, 1))  # view na maioria dos casos
                if src.dtype != self._view.dtype:
                    src = src.astype(self._view.dtype, copy=False)
                self._view[...] = src
                return

        raise ValueError(f"obs shape {arr.shape} != shm shape {self._view.shape}")

    def write(self, obs: Any) -> None:
        if self._view is None:
            return
        self._write_compatible(obs)
        self._obs_seq += 1

    def close(self) -> None:
        try:
            if self._shm is not None:
                self._shm.close()
        except Exception:
            pass
        self._shm = None
        self._view = None
        self._obs_seq = 0


# ----------------------------------------------------------------------
# Main actor loop
# ----------------------------------------------------------------------
def actor_loop(env: DoomDMEnv, conn: Connection) -> None:
    print("[ACTOR] Loop iniciado.", flush=True)

    shm_writer = ShmObsWriter()
    step_count = 0

    try:
        while True:
            try:
                msg = conn.recv()
            except EOFError:
                break

            if msg is None:
                break

            cmd = msg.get("cmd")

            if cmd == "get_spaces":
                conn.send({"obs_space": env.observation_space, "action_space": env.action_space})

            elif cmd == "set_shm_obs":
                # Esperado:
                #   {cmd, shm_name=<str>, shape=[...], dtype="uint8"}
                # Compat:
                #   aceita também "name" no lugar de "shm_name"
                try:
                    shm_name = msg.get("shm_name") or msg.get("name")
                    if not shm_name:
                        raise KeyError("missing shm_name (or name)")

                    shape = tuple(int(x) for x in msg["shape"])
                    dtype = np.dtype(msg["dtype"])

                    shm_writer.attach(shm_name=str(shm_name), shape=shape, dtype=dtype)
                    conn.send({"ok": True})
                except Exception as e:
                    # Não derruba o ator; apenas desabilita SHM e segue com IPC normal.
                    print(f"[ACTOR][ERROR] set_shm_obs: {e!r}", flush=True)
                    try:
                        shm_writer.close()
                    except Exception:
                        pass
                    conn.send({"ok": False, "error": str(e)})

            elif cmd == "reset":
                result = handle_reset(env)
                if "error" in result:
                    conn.send(result)
                    continue

                obs = result["obs"]
                info = result.get("info", {})

                if shm_writer.enabled:
                    try:
                        shm_writer.write(obs)
                        conn.send({"info": info, "obs_seq": shm_writer.obs_seq})
                    except Exception as e:
                        conn.send({"error": f"shm_write_reset_failed: {e}"})
                else:
                    conn.send({"obs": obs, "info": info})

            elif cmd == "step":
                step_count += 1
                if step_count % 10000 == 0:
                    print(f"[ACTOR] Steps: {step_count}", flush=True)

                result = handle_step(env, int(msg.get("action", 0)))
                if "error" in result:
                    conn.send(result)
                    continue

                obs = result["obs"]
                reward = result["reward"]
                done = result["done"]
                info = result.get("info", {})

                if shm_writer.enabled:
                    try:
                        shm_writer.write(obs)
                        conn.send(
                            {
                                "reward": reward,
                                "done": done,
                                "info": info,
                                "obs_seq": shm_writer.obs_seq,
                            }
                        )
                    except Exception as e:
                        conn.send({"error": f"shm_write_step_failed: {e}"})
                else:
                    conn.send(result)

            elif cmd == "close":
                try:
                    env.close()
                except Exception:
                    pass
                try:
                    shm_writer.close()
                except Exception:
                    pass
                try:
                    conn.send({"ok": True})
                except Exception:
                    pass
                break

            else:
                conn.send({"error": f"cmd inválido: {cmd!r}"})

    finally:
        try:
            env.close()
        except Exception:
            pass
        try:
            shm_writer.close()
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass
        print("[ACTOR] Encerrado.", flush=True)


# ----------------------------------------------------------------------
# Entrypoint
# ----------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    try:
        print("[ACTOR] Carregando VizDoom antes de conectar ao trainer...", flush=True)
        env = make_env(args)
    except Exception as e:
        print(f"[ACTOR][FATAL] Falha ao criar env: {e!r}", flush=True)
        traceback.print_exc()
        raise

    address = (args.trainer_host, args.trainer_port)
    print(f"[ACTOR] Conectando em {address}...", flush=True)

    conn: Optional[Connection] = None
    for _ in range(30):
        try:
            conn = Client(address, authkey=args.auth_key.encode("utf-8"))
            break
        except ConnectionRefusedError:
            time.sleep(1.0)

    if conn is None:
        try:
            env.close()
        except Exception:
            pass
        raise RuntimeError("Falha ao conectar no Treinador.")

    print("[ACTOR] Conectado (TCP). Entrando no loop.", flush=True)

    actor_loop(env, conn)


if __name__ == "__main__":
    main()

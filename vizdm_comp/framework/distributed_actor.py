#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from multiprocessing import shared_memory
from multiprocessing.connection import Client, Connection
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import yaml
import sys

# Ajuste conforme a estrutura do seu projeto
from .config import (
    DMConfig,
    AgentConfig,
    EngineRewardConfig,
    ShapingConfig,
    PolicyConfig,
    RenderSettingsConfig,
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
    with open(yaml_path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f) or {}

    # 1. Ler Engine Reward
    reward_block = y.get("reward", {})
    if "engine" in reward_block:
        er_data = reward_block["engine"]
    else:
        er_data = y.get("engine_reward", {})
    er = EngineRewardConfig(**er_data)

    # 2. Ler Shaping
    if "shaping" in reward_block:
        sh_data = reward_block["shaping"]
    else:
        sh_data = y.get("shaping", {})
    sh = ShapingConfig(**sh_data)

    # 3. Ler Policy
    pol = PolicyConfig(**y.get("policy", {}))

    # 4. Ler Render Settings
    rs_data = y.get("render_settings", {})
    rs = RenderSettingsConfig(**rs_data)

    # Monta o dicionário de argumentos para o AgentConfig
    kwargs: Dict[str, Any] = dict(
        name=y.get("name", "Client"),
        colorset=y.get("colorset", 3),
        render_settings=rs,
        engine_reward=er,
        shaping=sh,
        policy=pol,
        model_dir=y.get("model_dir", "models"),
        model_name=y.get("model_name", "agent.zip"),
        train=bool(y.get("train", False)),
        train_steps=int(y.get("train_steps", 300_000)),
        # stack_frames=... REMOVIDO: AgentConfig original não suporta isso.
        # O trainer usará o valor default da CLI (--stack).
    )

    # Campos opcionais (compatibilidade)
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
    parser.add_argument(
        "--wad",
        default=None,
        help="WAD/PK3 extra (nome em framework/maps/ ou caminho completo).",
    )
    
    parser.add_argument(
        "--game-config", 
        default=None, 
        help="Caminho do arquivo .cfg do jogo (ex: tag.cfg)"
    )

    # Match settings
    parser.add_argument("--timelimit", type=float, default=0.0)
    parser.add_argument("--render", action="store_true")
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
    try:
        agent_cfg = load_agent_cfg_light(args.cfg)
    except TypeError as e:
        print(f"[ACTOR][FATAL] Erro ao instanciar AgentConfig: {e}")
        print("[ACTOR] DICA: Verifique se o vizdm_comp/framework/config.py mudou.")
        raise e

    print(
        f"[ACTOR] Agente: {agent_cfg.name} | "
        f"Map: {args.map} | WAD: {args.wad} | Config: {args.game_config} | "
        f"Render: {bool(args.render)} | Host: {bool(args.is_host)}",
        flush=True,
    )

    dm_cfg = DMConfig(
        total_players=args.players,
        port=args.port,
        join_ip=args.join_ip,
        map_name=str(args.map),
        wad=args.wad,
        config_file=args.game_config,
        timelimit_minutes=float(args.timelimit),
        render=bool(args.render),
    )

    print(f"[ACTOR] Criando DoomDMEnv (is_host={args.is_host})...", flush=True)
    env = DoomDMEnv(
        name=agent_cfg.name,
        is_host=bool(args.is_host),
        dm=dm_cfg,
        agent_config=agent_cfg,
    )
    print("[ACTOR] DoomDMEnv criado.", flush=True)
    return env


# ----------------------------------------------------------------------
# Step/reset handlers
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
# Shared-memory writer
# ----------------------------------------------------------------------
class ShmObsWriter:
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
        try:
            shm = shared_memory.SharedMemory(name=shm_name, create=False)
        except FileNotFoundError:
             raise ValueError(f"Memória Compartilhada não encontrada: {shm_name}")
             
        view = np.ndarray(shape, dtype=dtype, buffer=shm.buf)

        if view.nbytes > shm.size:
            shm.close()
            raise ValueError(f"SHM pequeno: view.nbytes={view.nbytes} > shm.size={shm.size}")

        self._shm = shm
        self._view = view
        self._obs_seq = 0

    def _write_compatible(self, obs: Any) -> None:
        assert self._view is not None
        arr = np.asarray(obs)

        if arr.shape == self._view.shape:
            if arr.dtype != self._view.dtype:
                arr = arr.astype(self._view.dtype, copy=False)
            self._view[...] = arr
            return

        if arr.ndim == 3 and arr.shape[-1] == 1 and self._view.ndim == 2:
            h, w, _ = arr.shape
            if self._view.shape == (h, w):
                src = arr[..., 0]
                if src.dtype != self._view.dtype:
                    src = src.astype(self._view.dtype, copy=False)
                self._view[...] = src
                return

        if arr.ndim == 3 and arr.shape[-1] == 1 and self._view.ndim == 3 and self._view.shape[0] == 1:
            h, w, _ = arr.shape
            if self._view.shape == (1, h, w):
                src = np.transpose(arr, (2, 0, 1)) 
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
                try:
                    shm_name = msg.get("shm_name") or msg.get("name")
                    if not shm_name:
                        raise KeyError("missing shm_name (or name)")

                    shape = tuple(int(x) for x in msg["shape"])
                    dtype = np.dtype(msg["dtype"])

                    shm_writer.attach(shm_name=str(shm_name), shape=shape, dtype=dtype)
                    conn.send({"ok": True})
                except Exception as e:
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
                if step_count % 5000 == 0:
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
                try: env.close() 
                except: pass
                try: shm_writer.close()
                except: pass
                try: conn.send({"ok": True})
                except: pass
                break

            else:
                conn.send({"error": f"cmd inválido: {cmd!r}"})

    finally:
        try: env.close()
        except: pass
        try: shm_writer.close()
        except: pass
        try: conn.close()
        except: pass
        print("[ACTOR] Encerrado.", flush=True)


# ----------------------------------------------------------------------
# Entrypoint
# ----------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    address = (args.trainer_host, args.trainer_port)
    print(f"[ACTOR] Conectando em {address}...", flush=True)

    conn: Optional[Connection] = None
    for _ in range(30):
        try:
            conn = Client(address, authkey=args.auth_key.encode("utf-8"))
            break
        except (ConnectionRefusedError, OSError):
            time.sleep(1.0)

    if conn is None:
        print("[ACTOR] FALHA CRÍTICA: Timeout conectando ao Treinador.", flush=True)
        sys.exit(1)

    print("[ACTOR] Conectado (TCP). Carregando VizDoom...", flush=True)
    try:
        env = make_env(args)
    except Exception as e:
        print(f"[ACTOR] Erro ao criar ENV: {e}", flush=True)
        try:
            conn.send({"error": str(e)})
        except Exception:
            pass
        sys.exit(1)

    actor_loop(env, conn)


if __name__ == "__main__":
    main()
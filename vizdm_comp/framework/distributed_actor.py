#!/usr/bin/env python3
from __future__ import annotations
from .env import DoomDMEnv 
import argparse
import time
from multiprocessing import shared_memory
from multiprocessing.connection import Client, Connection
from typing import Any, Dict, Optional, Tuple

import numpy as np
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


# ----------------------------------------------------------------------
# Compat helpers
# ----------------------------------------------------------------------
def _agentconfig_supports(field_name: str) -> bool:
    """Compat: só passa kwargs que existirem no AgentConfig atual."""
    try:
        return field_name in getattr(AgentConfig, "__dataclass_fields__", {})
    except Exception:
        return False


def load_agent_cfg_light(cfg_path: str) -> AgentConfig:
    """
    Carrega o YAML e converte dicionários aninhados nas DataClasses corretas.
    Resolve o problema da chave 'reward' e 'render_settings'.
    """
    import yaml # Garante que yaml está importado
    
    with open(cfg_path, "r") as f:
        data = yaml.safe_load(f)

    # --- CORREÇÃO 1: Tratamento da chave 'reward' ---
    # Se o YAML tem 'reward', nós extraímos 'engine' e 'shaping' de dentro dele
    if "reward" in data:
        reward_data = data.pop("reward") # Remove a chave 'reward'
        
        # Converte o dicionário 'engine' em objeto EngineRewardConfig
        if "engine" in reward_data:
            data["engine_reward"] = EngineRewardConfig(**reward_data["engine"])
            
        # Converte o dicionário 'shaping' em objeto ShapingConfig
        if "shaping" in reward_data:
            data["shaping"] = ShapingConfig(**reward_data["shaping"])

    # --- CORREÇÃO 2: Tratamento da chave 'render_settings' ---
    # Se tiver render_settings como dicionário, converte para objeto
    if "render_settings" in data and isinstance(data["render_settings"], dict):
        data["render_settings"] = RenderSettingsConfig(**data["render_settings"])

    # --- CORREÇÃO 3: Tratamento de Policy e Engine (caso venham como dict) ---
    # O Python não converte dict para dataclass automaticamente aninhado
    if "engine_reward" in data and isinstance(data["engine_reward"], dict):
        data["engine_reward"] = EngineRewardConfig(**data["engine_reward"])
        
    if "shaping" in data and isinstance(data["shaping"], dict):
        data["shaping"] = ShapingConfig(**data["shaping"])

    # Agora criamos o AgentConfig com os dados limpos
    return AgentConfig(**data)

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
    
    parser.add_argument("--game-config", default=None, help="Caminho do .cfg do jogo (tag.cfg)")

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
def make_env(args):
    # Carrega configurações do agente
    agent_cfg = load_agent_cfg_light(args.cfg)
    
    # Lógica de renderização (Host vê, clientes não)
    render_mode = False
    if args.render:
        if args.is_host:
            render_mode = True
    
    # Cria o objeto de configuração do Deathmatch
    dm_cfg = DMConfig(
        config_file=args.game_config,
        wad=args.wad,
        map_name=args.map,
        timelimit_minutes=args.timelimit if args.timelimit > 0 else 3.0,
        render=render_mode,
        total_players=args.players
    )


    # --- A CORREÇÃO ESTÁ AQUI ---
    # Agora passamos os 4 argumentos que o env.py exige, com os nomes certos.
    env = DoomDMEnv(
        name=agent_cfg.name,      # O nome vem do YAML do agente
        is_host=args.is_host,     # Se sou host ou cliente
        dm=dm_cfg,                # O env.py espera 'dm', não 'dm_config'
        agent_config=agent_cfg    # O env.py espera 'agent_config'
    )
    # ----------------------------
    
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

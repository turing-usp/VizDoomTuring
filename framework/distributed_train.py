#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import csv
import os
import subprocess
import sys
import time
from dataclasses import replace
from multiprocessing.connection import Connection, Listener, wait
from typing import Any, Dict, List, Set, Tuple, Optional

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv, VecFrameStack, VecTransposeImage

from .client import load_agent_cfg
from .config import AgentConfig
from .policies import build_sb3, resolve_algo

try:
    # Python 3.8+
    from multiprocessing.shared_memory import SharedMemory
except Exception:  # pragma: no cover
    SharedMemory = None  # type: ignore


# ============================================================
# Shared memory helper (observations)
# ============================================================

class SharedObsManager:
    """
    Owns one SharedMemory segment per actor, each holding a single observation
    buffer shaped like obs_space.shape (uint8).
    """

    def __init__(self, num_envs: int, obs_shape: Tuple[int, ...], dtype: np.dtype):
        if SharedMemory is None:
            raise RuntimeError("multiprocessing.shared_memory não disponível nesta versão de Python.")
        if dtype != np.uint8:
            raise ValueError(f"SharedObsManager requer dtype uint8, recebido: {dtype}")

        self.num_envs = int(num_envs)
        self.obs_shape = tuple(int(x) for x in obs_shape)
        self.dtype = np.uint8

        nbytes = int(np.prod(self.obs_shape)) * np.dtype(self.dtype).itemsize
        if nbytes <= 0:
            raise ValueError(f"obs_shape inválido: {self.obs_shape}")

        self._shms: List[SharedMemory] = []
        self.views: List[np.ndarray] = []

        for _ in range(self.num_envs):
            shm = SharedMemory(create=True, size=nbytes)
            view = np.ndarray(self.obs_shape, dtype=self.dtype, buffer=shm.buf)
            view.fill(0)
            self._shms.append(shm)
            self.views.append(view)

    @property
    def names(self) -> List[str]:
        return [s.name for s in self._shms]

    def close(self) -> None:
        # Order matters on Windows: close then unlink.
        for shm in self._shms:
            try:
                shm.close()
            except Exception:
                pass
        for shm in self._shms:
            try:
                shm.unlink()
            except Exception:
                pass
        self._shms.clear()
        self.views.clear()


# ============================================================
# Remote VecEnv (optional SHM obs)
# ============================================================

class RemoteDMVecEnv(VecEnv):
    """
    VecEnv robusto que delega reset/step para N atores remotos via IPC.
    Opcional: shared memory para observações (reduz IPC/cópias).

    IMPORTANT:
      Para compatibilidade com actors antigos que esperavam msg["name"],
      o handshake envia *ambos* "shm_name" e "name".
    """

    def __init__(
        self,
        conns: List[Connection],
        obs_space: spaces.Space,
        action_space: spaces.Space,
        *,
        shm_obs: bool = False,
    ):
        if not conns:
            raise ValueError("É necessário pelo menos 1 conexão de ator")

        self._conns: List[Connection] = conns
        self._waiting: bool = False
        self.num_actors = len(conns)

        self._use_shm: bool = False
        self._shm: Optional[SharedObsManager] = None
        self._step_async_started_at: Optional[float] = None
        self._last_reply_span_s: float = 0.0
        self._perf_last_log_at: float = time.time()
        self._perf_step_calls: int = 0
        self._perf_env_steps: int = 0
        self._perf_done_count: int = 0
        self._perf_wait_total_s: float = 0.0
        self._perf_wait_max_s: float = 0.0
        self._perf_batch_total_s: float = 0.0
        self._perf_batch_max_s: float = 0.0
        self._perf_reply_span_total_s: float = 0.0
        self._perf_reply_span_max_s: float = 0.0
        self._perf_csv_path: Optional[str] = None
        self._perf_csv_header_written: bool = False

        super().__init__(
            num_envs=len(conns),
            observation_space=obs_space,
            action_space=action_space,
        )
        print(f"[TRAIN] RemoteDMVecEnv criado com {self.num_envs} envs (jogadores).")

        if shm_obs:
            self._try_enable_shm_obs()

    def _log_perf_window_if_due(self) -> None:
        now = time.time()
        elapsed = now - self._perf_last_log_at
        if elapsed < 5.0 or self._perf_step_calls <= 0:
            return

        step_calls = self._perf_step_calls
        env_steps = self._perf_env_steps
        avg_wait_ms = (self._perf_wait_total_s / step_calls) * 1000.0
        max_wait_ms = self._perf_wait_max_s * 1000.0
        avg_batch_ms = (self._perf_batch_total_s / step_calls) * 1000.0
        max_batch_ms = self._perf_batch_max_s * 1000.0
        avg_reply_span_ms = (self._perf_reply_span_total_s / step_calls) * 1000.0
        max_reply_span_ms = self._perf_reply_span_max_s * 1000.0
        vec_steps_per_s = step_calls / elapsed
        env_steps_per_s = env_steps / elapsed

        status = "ok"
        if avg_batch_ms > 500.0 or max_reply_span_ms > 1000.0:
            status = "warn"
        if avg_batch_ms > 1000.0 or max_reply_span_ms > 3000.0:
            status = "critical"

        print(
            f"[TRAIN][PERF][{status}] window={elapsed:.1f}s "
            f"vec_steps/s={vec_steps_per_s:.2f} env_steps/s={env_steps_per_s:.2f} "
            f"actor_decisions/s={vec_steps_per_s:.2f} "
            f"step_wait_ms(avg/max)={avg_wait_ms:.1f}/{max_wait_ms:.1f} "
            f"batch_ms(avg/max)={avg_batch_ms:.1f}/{max_batch_ms:.1f} "
            f"reply_span_ms(avg/max)={avg_reply_span_ms:.1f}/{max_reply_span_ms:.1f} "
            f"dones={self._perf_done_count}",
            flush=True,
        )

        if self._perf_csv_path:
            os.makedirs(os.path.dirname(self._perf_csv_path), exist_ok=True)
            row = {
                "window_s": round(elapsed, 6),
                "vec_steps_per_s": round(vec_steps_per_s, 6),
                "env_steps_per_s": round(env_steps_per_s, 6),
                "actor_decisions_per_s": round(vec_steps_per_s, 6),
                "step_wait_ms_avg": round(avg_wait_ms, 6),
                "step_wait_ms_max": round(max_wait_ms, 6),
                "batch_ms_avg": round(avg_batch_ms, 6),
                "batch_ms_max": round(max_batch_ms, 6),
                "reply_span_ms_avg": round(avg_reply_span_ms, 6),
                "reply_span_ms_max": round(max_reply_span_ms, 6),
                "dones": int(self._perf_done_count),
                "status": status,
            }
            write_header = (not self._perf_csv_header_written) and (not os.path.exists(self._perf_csv_path))
            with open(self._perf_csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
            self._perf_csv_header_written = True

        self._perf_last_log_at = now
        self._perf_step_calls = 0
        self._perf_env_steps = 0
        self._perf_done_count = 0
        self._perf_wait_total_s = 0.0
        self._perf_wait_max_s = 0.0
        self._perf_batch_total_s = 0.0
        self._perf_batch_max_s = 0.0
        self._perf_reply_span_total_s = 0.0
        self._perf_reply_span_max_s = 0.0

    def set_perf_csv_path(self, perf_csv_path: str) -> None:
        self._perf_csv_path = perf_csv_path

    def _wait_for_replies(
        self,
        pendentes: Set[int],
        timeout_total: float,
        context: str,
    ) -> Dict[int, Any]:
        results: Dict[int, Any] = {}
        t0 = time.time()
        last_log = time.time()
        conn_map = {self._conns[i]: i for i in pendentes}
        max_wait_handles = 60
        first_reply_at: Optional[float] = None
        last_reply_at: Optional[float] = None

        def wait_chunked(conns: List[Connection], timeout: float) -> List[Connection]:
            if len(conns) <= max_wait_handles:
                return list(wait(conns, timeout=timeout))

            ready: List[Connection] = []
            deadline = time.time() + timeout

            while True:
                for start in range(0, len(conns), max_wait_handles):
                    batch = conns[start:start + max_wait_handles]
                    ready.extend(wait(batch, timeout=0.0))

                if ready:
                    return ready

                if time.time() >= deadline:
                    return []

                time.sleep(0.01)

        while pendentes:
            if time.time() - t0 > timeout_total:
                raise TimeoutError(
                    f"[TRAIN] TIMEOUT CRÍTICO ({timeout_total}s) em {context}. "
                    f"Atores que não responderam: {sorted(list(pendentes))}."
                )

            if time.time() - last_log > 5.0:
                print(
                    f"[TRAIN][DEBUG] {context}: aguardando {len(pendentes)} atores: {list(pendentes)[:5]}...",
                    flush=True,
                )
                last_log = time.time()

            conns_to_wait = [self._conns[i] for i in pendentes]
            ready_conns = wait_chunked(conns_to_wait, timeout=0.1)

            for conn in ready_conns:
                idx = conn_map.get(conn)
                if idx is None or idx not in pendentes:
                    continue

                try:
                    msg = conn.recv()
                    if isinstance(msg, dict) and "error" in msg:
                        raise RuntimeError(f"[TRAIN] Ator {idx} reportou erro: {msg['error']}")
                    now = time.time()
                    if first_reply_at is None:
                        first_reply_at = now
                    last_reply_at = now
                    results[idx] = msg
                    pendentes.remove(idx)
                except EOFError:
                    raise RuntimeError(
                        f"[TRAIN] Conexão com ator {idx} fechada inesperadamente em {context}."
                    )
                except Exception as e:
                    print(f"[TRAIN][ERROR] Exceção ao ler ator {idx} em {context}: {e!r}", flush=True)
                    raise

        if first_reply_at is not None and last_reply_at is not None:
            self._last_reply_span_s = max(0.0, last_reply_at - first_reply_at)
        else:
            self._last_reply_span_s = 0.0

        return results

    def _try_enable_shm_obs(self) -> None:
        """
        Negocia SHM com os atores. Se qualquer ator não suportar,
        faz fallback para IPC normal sem quebrar.
        """
        if SharedMemory is None:
            print("[TRAIN][WARN] --shm-obs solicitado, mas SharedMemory não disponível. Ignorando.")
            return

        if not isinstance(self.observation_space, spaces.Box):
            print("[TRAIN][WARN] --shm-obs requer observation_space Box. Ignorando.")
            return

        obs_shape = tuple(int(x) for x in self.observation_space.shape)
        obs_dtype = np.dtype(self.observation_space.dtype)

        if obs_dtype != np.uint8:
            print(f"[TRAIN][WARN] --shm-obs requer obs uint8. Obs dtype={obs_dtype}. Ignorando.")
            return

        shm: Optional[SharedObsManager] = None
        try:
            shm = SharedObsManager(self.num_envs, obs_shape, dtype=np.uint8)

            # Envia setup para cada ator
            for i, conn in enumerate(self._conns):
                # Compat: alguns actors antigos esperavam a chave "name"
                conn.send(
                    {
                        "cmd": "set_shm_obs",
                        "shm_name": shm.names[i],
                        "name": shm.names[i],          # <-- compat com actor antigo
                        "shape": list(obs_shape),
                        "dtype": "uint8",
                    }
                )

            # Aguarda ACK de todos
            msgs = self._wait_for_replies(
                pendentes=set(range(self.num_envs)),
                timeout_total=30.0,
                context="set_shm_obs",
            )

            # Aceita ACKs diferentes:
            # - {"ok": True}
            # - {} / qualquer dict sem "ok" explícito
            # - "ok" / None (tolerante)
            for i in range(self.num_envs):
                m = msgs.get(i)
                if isinstance(m, dict):
                    if m.get("ok") is False:
                        raise RuntimeError(f"Ator {i} recusou SHM: {m!r}")
                # se não for dict, assume sucesso (compat)

            self._shm = shm
            self._use_shm = True
            print("[TRAIN] SHM obs habilitado: observações serão lidas via shared memory.")

        except Exception as e:
            print(f"[TRAIN][WARN] Falha ao habilitar SHM obs, fallback para IPC normal: {e!r}")
            try:
                if shm is not None:
                    shm.close()
            except Exception:
                pass
            self._shm = None
            self._use_shm = False

    def _read_obs_from_shm(self, idx: int) -> np.ndarray:
        assert self._shm is not None
        # Copia para evitar ser sobrescrito pelo próximo step/reset
        return np.array(self._shm.views[idx], copy=True)

    def reset(self) -> np.ndarray:
        print(f"[TRAIN] Enviando comando 'reset' para {self.num_envs} atores...", flush=True)
        for idx, conn in enumerate(self._conns):
            try:
                conn.send({"cmd": "reset"})
            except Exception as e:
                print(f"[TRAIN][ERROR] Falha ao enviar reset para ator {idx}: {e!r}")
                raise

        msgs = self._wait_for_replies(
            pendentes=set(range(self.num_envs)),
            timeout_total=120.0,
            context="reset()",
        )

        # Monta batch
        if self._use_shm:
            obs_shape = self.observation_space.shape
            obs_batch = np.empty((self.num_envs,) + obs_shape, dtype=np.uint8)
            for i in range(self.num_envs):
                obs_batch[i] = self._read_obs_from_shm(i)
            print(f"[TRAIN] reset() concluído (SHM). Shape={obs_batch.shape}", flush=True)
            return obs_batch

        obs_list: List[Any] = []
        for idx in range(self.num_envs):
            msg = msgs[idx]
            if not isinstance(msg, dict) or "obs" not in msg:
                raise RuntimeError(f"[TRAIN] Resposta inválida reset ator {idx}: {msg!r}")
            obs_list.append(msg["obs"])

        shapes = [np.asarray(o).shape for o in obs_list]
        if len(set(shapes)) != 1:
            raise RuntimeError(f"[TRAIN] Inconsistent obs shapes from actors: {list(enumerate(shapes))}")

        obs_batch = np.stack(obs_list, axis=0)
        print(f"[TRAIN] reset() concluído (IPC). Shape={obs_batch.shape}", flush=True)
        return obs_batch

    def step_async(self, actions):
        actions = np.array(actions).reshape((self.num_envs,))
        self._step_async_started_at = time.time()
        for idx, (conn, act) in enumerate(zip(self._conns, actions)):
            try:
                conn.send({"cmd": "step", "action": int(act)})
            except Exception as e:
                print(f"[TRAIN][ERROR] Falha ao enviar step para ator {idx}: {e!r}")
                raise
        self._waiting = True

    def step_wait(self):
        wait_started_at = time.time()
        msgs = self._wait_for_replies(
            pendentes=set(range(self.num_envs)),
            timeout_total=60.0,
            context="step_wait()",
        )
        self._waiting = False
        wait_elapsed_s = time.time() - wait_started_at
        batch_elapsed_s = (
            time.time() - self._step_async_started_at
            if self._step_async_started_at is not None
            else wait_elapsed_s
        )

        obs_shape = self.observation_space.shape
        if self._use_shm:
            obs_batch = np.empty((self.num_envs,) + obs_shape, dtype=np.uint8)
        else:
            obs_list: List[Any] = []

        rewards = np.empty((self.num_envs,), dtype=np.float32)
        dones = np.empty((self.num_envs,), dtype=bool)
        infos: List[Dict[str, Any]] = []

        for idx in range(self.num_envs):
            msg = msgs[idx]
            if not isinstance(msg, dict):
                raise RuntimeError(f"[TRAIN] Msg inválida ator {idx}: {msg!r}")

            rew = float(msg["reward"])
            done = bool(msg["done"])
            info = dict(msg.get("info", {}))

            if self._use_shm:
                obs = self._read_obs_from_shm(idx)
            else:
                obs = msg["obs"]

            if done:
                info["terminal_observation"] = np.array(obs, copy=True)

                # reset automático após done
                try:
                    self._conns[idx].send({"cmd": "reset"})
                    res = self._wait_for_replies(
                        pendentes={idx},
                        timeout_total=60.0,
                        context=f"reset_pos_done({idx})",
                    )
                    if self._use_shm:
                        obs = self._read_obs_from_shm(idx)
                    else:
                        obs = res[idx]["obs"]
                except Exception as e:
                    raise RuntimeError(f"[TRAIN] Erro no reset automático do ator {idx}: {e!r}")

            if self._use_shm:
                obs_batch[idx] = obs
            else:
                obs_list.append(obs)

            rewards[idx] = rew
            dones[idx] = done
            infos.append(info)

        self._perf_step_calls += 1
        self._perf_env_steps += self.num_envs
        self._perf_done_count += int(np.count_nonzero(dones))
        self._perf_wait_total_s += wait_elapsed_s
        self._perf_wait_max_s = max(self._perf_wait_max_s, wait_elapsed_s)
        self._perf_batch_total_s += batch_elapsed_s
        self._perf_batch_max_s = max(self._perf_batch_max_s, batch_elapsed_s)
        self._perf_reply_span_total_s += self._last_reply_span_s
        self._perf_reply_span_max_s = max(self._perf_reply_span_max_s, self._last_reply_span_s)
        self._log_perf_window_if_due()

        if self._use_shm:
            return obs_batch, rewards, dones, infos

        shapes = [np.asarray(o).shape for o in obs_list]
        if len(set(shapes)) != 1:
            raise RuntimeError(f"[TRAIN] Inconsistent obs shapes in step_wait: {list(enumerate(shapes))}")

        return np.stack(obs_list, axis=0), rewards, dones, infos

    def close(self):
        print("[TRAIN] Fechando RemoteDMVecEnv, enviando 'close' para atores...")
        if self._waiting:
            for conn in self._conns:
                try:
                    if conn.poll(0.1):
                        conn.recv()
                except Exception:
                    pass
            self._waiting = False

        for idx, conn in enumerate(self._conns):
            try:
                conn.send({"cmd": "close"})
                if conn.poll(2.0):
                    conn.recv()
            except Exception:
                pass
            try:
                conn.close()
            except Exception:
                pass

        if self._shm is not None:
            try:
                self._shm.close()
            except Exception:
                pass
            self._shm = None
            self._use_shm = False

    def render(self, mode: str = "human"):
        return None

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs):
        raise NotImplementedError()

    def env_is_wrapped(self, wrapper_class, indices=None) -> List[bool]:
        return [False] * self.num_actors

    def get_attr(self, attr_name: str, indices=None) -> List[Any]:
        return [None] * self.num_actors

    def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
        pass


# ============================================================
# Debug callback
# ============================================================

class DebugCallback(BaseCallback):
    def __init__(
        self,
        log_every: int = 1_000,
        reward_window: int = 10_000,
        metrics_csv_path: Optional[str] = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.log_every = log_every
        self.reward_window = reward_window
        self.metrics_csv_path = metrics_csv_path
        self._last = 0
        self._recent_rewards = collections.deque(maxlen=reward_window)
        self._rollout_start_time: float | None = None
        self._train_started_at: float | None = None
        self._csv_header_written: bool = False

    def _append_metrics_row(self, row: Dict[str, Any]) -> None:
        if not self.metrics_csv_path:
            return

        os.makedirs(os.path.dirname(self.metrics_csv_path), exist_ok=True)
        fieldnames = list(row.keys())
        write_header = (not self._csv_header_written) and (not os.path.exists(self.metrics_csv_path))
        with open(self.metrics_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        self._csv_header_written = True

    def _on_training_start(self) -> None:
        self._train_started_at = time.time()
        print("[DEBUG] == TRAINING START ==")

    def _on_rollout_start(self) -> None:
        self._rollout_start_time = time.time()
        print(f"[DEBUG] ==== ROLLOUT START (num_timesteps={self.num_timesteps}) ====")

    def _on_rollout_end(self) -> None:
        now = time.time()
        dt = None
        if self._rollout_start_time is not None:
            dt = now - self._rollout_start_time
        logger_values = dict(getattr(getattr(self.model, "logger", None), "name_to_value", {}))
        time_elapsed = None
        if self._train_started_at is not None:
            time_elapsed = now - self._train_started_at
        fps = None
        if time_elapsed and time_elapsed > 0:
            fps = float(self.num_timesteps) / float(time_elapsed)
        mean_reward_window = (
            float(sum(self._recent_rewards) / len(self._recent_rewards))
            if self._recent_rewards
            else float("nan")
        )

        print(
            f"[DEBUG] ==== ROLLOUT END (num_timesteps={self.num_timesteps}) "
            + (f"duracao={dt:.2f}s" if dt is not None else "")
        )

        row: Dict[str, Any] = {
            "wall_time_s": round(time_elapsed, 6) if time_elapsed is not None else "",
            "num_timesteps": int(self.num_timesteps),
            "rollout_duration_s": round(dt, 6) if dt is not None else "",
            "fps": round(fps, 6) if fps is not None else "",
            "reward_mean_window": round(mean_reward_window, 6) if np.isfinite(mean_reward_window) else "",
            "train_learning_rate": logger_values.get("train/learning_rate", ""),
            "train_entropy_loss": logger_values.get("train/entropy_loss", ""),
            "train_policy_gradient_loss": logger_values.get("train/policy_gradient_loss", ""),
            "train_value_loss": logger_values.get("train/value_loss", ""),
            "train_loss": logger_values.get("train/loss", ""),
            "train_approx_kl": logger_values.get("train/approx_kl", ""),
            "train_clip_fraction": logger_values.get("train/clip_fraction", ""),
            "train_explained_variance": logger_values.get("train/explained_variance", ""),
            "train_std": logger_values.get("train/std", ""),
            "rollout_ep_rew_mean": logger_values.get("rollout/ep_rew_mean", ""),
            "rollout_ep_len_mean": logger_values.get("rollout/ep_len_mean", ""),
            "time_fps_logger": logger_values.get("time/fps", ""),
            "time_iterations": logger_values.get("time/iterations", ""),
            "time_total_timesteps_logger": logger_values.get("time/total_timesteps", ""),
        }
        self._append_metrics_row(row)

    def _on_training_end(self) -> None:
        print("[DEBUG] == TRAINING END ==")

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", None)
        if rewards is not None:
            for r in np.asarray(rewards).ravel():
                self._recent_rewards.append(float(r))

        if self.num_timesteps - self._last >= self.log_every:
            self._last = self.num_timesteps
            mean_r = float(sum(self._recent_rewards) / len(self._recent_rewards)) if self._recent_rewards else float("nan")
            print(
                f"[DEBUG] num_timesteps={self.num_timesteps}, "
                f"mean_reward_window={mean_r:.3f} (últimos {len(self._recent_rewards)} passos)"
            )
        return True


# ============================================================
# CLI / orchestration
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treinador distribuído VizDoom DM")
    parser.add_argument("--cfg", required=True, help="YAML do agente base")
    parser.add_argument("--num-matches", type=int, default=None)
    parser.add_argument("--actors-per-match", type=int, default=None)
    parser.add_argument("--num-actors", type=int, default=4)
    parser.add_argument("--game-port", type=int, default=5029)
    parser.add_argument("--game-ip", default="127.0.0.1")
    parser.add_argument("--timelimit", type=float, default=0.0)
    parser.add_argument("--stack", type=int, default=4)
    parser.add_argument("--render-host", action="store_true")
    parser.add_argument("--render-host-agent", action="store_true")
    parser.add_argument("--render-all", action="store_true")

    parser.add_argument("--map", default="map01", help="Nome do mapa (ex: MAP01 / map01)")
    parser.add_argument("--scenario", default=None, help="Cenário base .cfg/.wad/.pk3")
    parser.add_argument("--wad", default=None, help="Arquivo WAD/PK3 (nome em framework/maps ou caminho)")
    parser.add_argument("--frame-skip", type=int, default=8)
    parser.add_argument("--ticrate", type=int, default=30)

    parser.add_argument("--trainer-host", default="127.0.0.1")
    parser.add_argument("--trainer-port", type=int, default=7000)
    parser.add_argument("--auth-key", default="vizdoom_dm")
    parser.add_argument("--host-start-delay", type=float, default=1.5)
    parser.add_argument("--actor-start-delay", type=float, default=0.05)

    parser.add_argument(
        "--shm-obs",
        action="store_true",
        help="Usa shared memory para observações (requer distributed_actor com cmd set_shm_obs).",
    )

    parser.add_argument(
        "--warmstart-reset-steps",
        action="store_true",
        help="Carrega pesos do checkpoint, mas reinicia steps/schedules/otimizador do zero.",
    )

    return parser.parse_args()


def log_torch_device_info() -> None:
    try:
        print(
            f"[TRAIN][GPU] torch={torch.__version__} cuda_available={torch.cuda.is_available()} "
            f"cuda_version={torch.version.cuda}",
            flush=True,
        )
        if torch.cuda.is_available():
            for idx in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(idx)
                total_gib = props.total_memory / (1024 ** 3)
                print(
                    f"[TRAIN][GPU] device[{idx}]={torch.cuda.get_device_name(idx)} "
                    f"vram={total_gib:.2f} GiB",
                    flush=True,
                )
    except Exception as e:
        print(f"[TRAIN][GPU][WARN] Falha ao coletar info da GPU: {e!r}", flush=True)


def normalize_topology(args: argparse.Namespace) -> None:
    if args.num_matches is not None and args.actors_per_match is not None:
        print(f"[TRAIN] Config manual: {args.num_matches} partidas x {args.actors_per_match} jogadores.")
        args.num_actors = args.num_matches * args.actors_per_match
    else:
        if args.num_matches is None:
            args.num_matches = 1
        if args.actors_per_match is None:
            args.actors_per_match = args.num_actors
        args.num_actors = args.num_matches * args.actors_per_match

    print(f"[TRAIN] Total atores calculados: {args.num_actors}")


def start_listener(args: argparse.Namespace) -> Tuple[Listener, Tuple[str, int]]:
    address = (args.trainer_host, args.trainer_port)
    print(f"[TRAIN] Iniciando Listener IPC em {address}...")
    listener = Listener(address, backlog=max(1, args.num_actors), authkey=args.auth_key.encode("utf-8"))
    return listener, address


def _build_actor_cmd(
    args: argparse.Namespace,
    module_name: str,
    cfg_path: str,
    port: int,
    is_host: bool,
    players: int,
) -> List[str]:
    cmd = [
        sys.executable,
        "-m",
        module_name,
        "--cfg",
        cfg_path,
        "--players",
        str(players),
        "--port",
        str(port),
        "--join-ip",
        args.game_ip,
        "--timelimit",
        str(args.timelimit),
        "--trainer-host",
        args.trainer_host,
        "--trainer-port",
        str(args.trainer_port),
        "--auth-key",
        args.auth_key,
        "--map",
        str(args.map),
        "--frame-skip",
        str(args.frame_skip),
        "--ticrate",
        str(args.ticrate),
    ]

    if args.scenario:
        cmd += ["--scenario", str(args.scenario)]
    if args.wad:
        cmd += ["--wad", str(args.wad)]

    if is_host:
        cmd.append("--is-host")
        if args.render_all or args.render_host:
            cmd.append("--render")
        if args.render_host_agent:
            cmd += ["--render", "--render-agent-view"]
    elif args.render_all:
        cmd.append("--render")

    return cmd


def launch_actors(args: argparse.Namespace, cfg_path: str) -> List[subprocess.Popen]:
    num_matches = args.num_matches
    actors_per_match = args.actors_per_match
    procs: List[subprocess.Popen] = []

    module_name = "framework.distributed_actor"

    spawn_delay_actor = max(0.0, float(args.actor_start_delay))
    spawn_delay_host = max(0.0, float(args.host_start_delay))

    print(f"[TRAIN] Lançando {num_matches} partidas... map={args.map!r} wad={args.wad!r}")

    for match_idx in range(num_matches):
        match_port = args.game_port + match_idx
        print(f"\n[TRAIN] === Partida {match_idx} (Porta {match_port}) ===")

        cmd_host = _build_actor_cmd(args, module_name, cfg_path, match_port, True, actors_per_match)
        print("[TRAIN] Lançando HOST...")
        print("[TRAIN][CMD-HOST]", " ".join(cmd_host))
        procs.append(subprocess.Popen(cmd_host))

    print(f"[TRAIN] Aguardando {spawn_delay_host}s para os hosts iniciarem...")
    time.sleep(spawn_delay_host)

    for local_idx in range(1, actors_per_match):
        for match_idx in range(num_matches):
            match_port = args.game_port + match_idx
            cmd_client = _build_actor_cmd(args, module_name, cfg_path, match_port, False, actors_per_match)
            print(f"[TRAIN] Lançando Cliente {local_idx}...")
            print("[TRAIN][CMD-CLI]", " ".join(cmd_client))
            procs.append(subprocess.Popen(cmd_client))
            if spawn_delay_actor > 0:
                time.sleep(spawn_delay_actor)

    return procs


def launch_actors_staggered(args: argparse.Namespace, cfg_path: str) -> List[subprocess.Popen]:
    num_matches = args.num_matches
    actors_per_match = args.actors_per_match
    procs: List[subprocess.Popen] = []

    module_name = "framework.distributed_actor"
    spawn_delay_actor = max(0.0, float(args.actor_start_delay))
    spawn_delay_host = max(0.0, float(args.host_start_delay))

    print(f"[TRAIN] LanÃ§ando {num_matches} partidas de forma escalonada... map={args.map!r} wad={args.wad!r}")

    for match_idx in range(num_matches):
        match_port = args.game_port + match_idx
        print(f"\n[TRAIN] === Partida {match_idx} (Porta {match_port}) ===")

        cmd_host = _build_actor_cmd(args, module_name, cfg_path, match_port, True, actors_per_match)
        print("[TRAIN] LanÃ§ando HOST...")
        print("[TRAIN][CMD-HOST]", " ".join(cmd_host))
        procs.append(subprocess.Popen(cmd_host))

        if spawn_delay_host > 0:
            print(f"[TRAIN] Aguardando {spawn_delay_host}s para Host da partida {match_idx} iniciar...")
            time.sleep(spawn_delay_host)

        for local_idx in range(1, actors_per_match):
            cmd_client = _build_actor_cmd(args, module_name, cfg_path, match_port, False, actors_per_match)
            print(f"[TRAIN] LanÃ§ando Cliente {local_idx} da partida {match_idx}...")
            print("[TRAIN][CMD-CLI]", " ".join(cmd_client))
            procs.append(subprocess.Popen(cmd_client))
            if spawn_delay_actor > 0:
                time.sleep(spawn_delay_actor)

    return procs


def accept_n_actor_conns(listener: Listener, expected: int, label: str) -> List[Connection]:
    print(f"[TRAIN] Aguardando {expected} conexÃµes para {label}...")
    conns: List[Connection] = []
    for idx in range(expected):
        try:
            conns.append(listener.accept())
        except Exception as e:
            raise RuntimeError(
                f"Falha ao aceitar conexÃ£o {idx+1}/{expected} de {label}: {e!r}"
            ) from e
        print(f"[TRAIN] {label}: {idx+1}/{expected} conectados.")
    return conns


def launch_actors_staggered_and_accept(
    args: argparse.Namespace,
    cfg_path: str,
    listener: Listener,
) -> Tuple[List[subprocess.Popen], List[Connection]]:
    num_matches = args.num_matches
    actors_per_match = args.actors_per_match
    procs: List[subprocess.Popen] = []
    conns: List[Connection] = []

    module_name = "framework.distributed_actor"
    spawn_delay_actor = max(0.0, float(args.actor_start_delay))
    spawn_delay_host = max(0.0, float(args.host_start_delay))

    print(f"[TRAIN] LanÃ§ando {num_matches} partidas de forma escalonada com handshake por partida... map={args.map!r} wad={args.wad!r}")

    for match_idx in range(num_matches):
        match_port = args.game_port + match_idx
        label = f"partida {match_idx}"
        print(f"\n[TRAIN] === Partida {match_idx} (Porta {match_port}) ===")

        cmd_host = _build_actor_cmd(args, module_name, cfg_path, match_port, True, actors_per_match)
        print("[TRAIN] LanÃ§ando HOST...")
        print("[TRAIN][CMD-HOST]", " ".join(cmd_host))
        procs.append(subprocess.Popen(cmd_host))

        if spawn_delay_host > 0:
            print(f"[TRAIN] Aguardando {spawn_delay_host}s para Host da partida {match_idx} iniciar...")
            time.sleep(spawn_delay_host)

        for local_idx in range(1, actors_per_match):
            cmd_client = _build_actor_cmd(args, module_name, cfg_path, match_port, False, actors_per_match)
            print(f"[TRAIN] LanÃ§ando Cliente {local_idx} da partida {match_idx}...")
            print("[TRAIN][CMD-CLI]", " ".join(cmd_client))
            procs.append(subprocess.Popen(cmd_client))
            if spawn_delay_actor > 0:
                time.sleep(spawn_delay_actor)

        conns.extend(accept_n_actor_conns(listener, actors_per_match, label))

    return procs, conns


def accept_actor_conns(listener: Listener, num_actors: int) -> List[Connection]:
    print(f"[TRAIN] Aguardando {num_actors} conexões (Socket)...")
    conns: List[Connection] = []
    for i in range(num_actors):
        try:
            conns.append(listener.accept())
        except Exception as e:
            raise RuntimeError(
                f"Falha ao aceitar conexÃ£o {i+1}/{num_actors}. "
                f"Algum ator provavelmente caiu durante a inicializaÃ§Ã£o: {e!r}"
            ) from e
        print(f"[TRAIN] {i+1}/{num_actors} conectados.")
    return conns


def fetch_spaces(conns: List[Connection]) -> Tuple[spaces.Space, spaces.Space]:
    print("[TRAIN] Solicitando spaces...")
    conns[0].send({"cmd": "get_spaces"})
    msg = conns[0].recv()
    return msg["obs_space"], msg["action_space"]


def build_vec_env(conns: List[Connection], stack: int, *, shm_obs: bool) -> VecEnv:
    obs_space, action_space = fetch_spaces(conns)
    base_env = RemoteDMVecEnv(conns, obs_space, action_space, shm_obs=shm_obs)
    env: VecEnv = VecTransposeImage(base_env)
    env = VecFrameStack(env, n_stack=int(stack), channels_order="first")
    return env


def _find_remote_base_env(env: Any) -> Optional[RemoteDMVecEnv]:
    cur = env
    visited = set()
    while cur is not None and id(cur) not in visited:
        visited.add(id(cur))
        if isinstance(cur, RemoteDMVecEnv):
            return cur
        cur = getattr(cur, "venv", None)
    return None


def auto_adjust_n_steps(
    agent_cfg: AgentConfig,
    env: VecEnv,
    max_rollout_gib: float = 4.0,
    safety_factor: float = 4.0,
) -> AgentConfig:
    import numpy as _np

    obs_shape = env.observation_space.shape
    n_envs = env.num_envs

    obs_elems = int(_np.prod(obs_shape))
    bytes_per_elem = 4
    target_bytes = max_rollout_gib * (1024**3)

    max_n_steps = int(target_bytes / (safety_factor * n_envs * obs_elems * bytes_per_elem))
    if max_n_steps <= 0:
        raise RuntimeError(
            f"[TRAIN][FATAL] Configuração inviável de memória: "
            f"num_envs={n_envs}, obs_shape={obs_shape}, "
            f"max_rollout_gib={max_rollout_gib}, safety_factor={safety_factor}."
        )

    lk = dict(agent_cfg.policy.learn_kwargs)
    current_n_steps = int(lk.get("n_steps", 2048))

    if current_n_steps > max_n_steps:
        print(
            "[TRAIN][WARN] Ajustando n_steps por limite de memória: "
            f"{current_n_steps} -> {max_n_steps} "
            f"(num_envs={n_envs}, obs_shape={obs_shape}, "
            f"max_rollout_gib={max_rollout_gib}, safety_factor={safety_factor})"
        )
        lk["n_steps"] = max_n_steps
        new_policy = replace(agent_cfg.policy, learn_kwargs=lk)
        agent_cfg = replace(agent_cfg, policy=new_policy)
    else:
        print(
            f"[TRAIN] n_steps={current_n_steps} mantido (max_n_steps={max_n_steps}, "
            f"num_envs={n_envs}, obs_shape={obs_shape})"
        )

    return agent_cfg


def build_model(
    agent_cfg: AgentConfig,
    env: VecEnv,
    save_path: str,
    *,
    warmstart_reset_steps: bool = False,
):
    algo_cls = resolve_algo(agent_cfg.policy.algo)
    print("[DEBUG] learn_kwargs:", agent_cfg.policy.learn_kwargs)
    if os.path.exists(save_path):
        if warmstart_reset_steps:
            print(f"[TRAIN] Warm start com reset de steps a partir de: {save_path}")
            try:
                loaded_model = algo_cls.load(save_path, env=env)
                fresh_model = build_sb3(
                    algo_cls,
                    "CnnPolicy",
                    env,
                    agent_cfg.policy.policy_kwargs,
                    agent_cfg.policy.learn_kwargs,
                )
                fresh_model.policy.load_state_dict(loaded_model.policy.state_dict(), strict=True)
                return fresh_model
            except Exception as e:
                print(f"[TRAIN] Erro no warm start: {e!r}. Caindo para carregamento normal.")

        print(f"[TRAIN] Carregando modelo: {save_path}")
        try:
            return algo_cls.load(save_path, env=env)
        except Exception as e:
            print(f"[TRAIN] Erro ao carregar: {e!r}. Criando novo.")

    print(f"[TRAIN] Criando novo modelo ({agent_cfg.policy.algo}).")
    return build_sb3(
        algo_cls,
        "CnnPolicy",
        env,
        agent_cfg.policy.policy_kwargs,
        agent_cfg.policy.learn_kwargs,
    )


def _set_chunk_lr(model, target_total_steps: int) -> None:
    lr_range = getattr(model, "_lr_range", None)
    if not lr_range:
        return

    max_lr, min_lr = lr_range
    total = max(int(target_total_steps), 1)
    t = int(getattr(model, "num_timesteps", 0))
    progress = max(0.0, min(1.0, t / total))
    lr = float(max_lr) - (float(max_lr) - float(min_lr)) * progress

    def _const_lr(_progress_remaining: float) -> float:
        return lr

    if hasattr(model, "lr_schedule"):
        model.lr_schedule = _const_lr

    opt = getattr(getattr(model, "policy", None), "optimizer", None)
    if opt is not None:
        for pg in opt.param_groups:
            pg["lr"] = lr

    try:
        model.learning_rate = lr
    except Exception:
        pass

    print(
        f"[TRAIN][LR] num_timesteps={t}/{total}, progress={progress:.4f}, lr_chunk={lr}",
        flush=True,
    )


def _set_chunk_ent_coef(model, target_total_steps: int) -> None:
    ent_range = getattr(model, "_ent_coef_range", None)
    if not ent_range:
        return

    max_ent, min_ent = ent_range
    total = max(int(target_total_steps), 1)
    t = int(getattr(model, "num_timesteps", 0))
    progress = max(0.0, min(1.0, t / total))
    ent_coef = float(max_ent) - (float(max_ent) - float(min_ent)) * progress

    try:
        model.ent_coef = ent_coef
    except Exception:
        pass

    print(
        f"[TRAIN][ENT] num_timesteps={t}/{total}, progress={progress:.4f}, ent_coef_chunk={ent_coef}",
        flush=True,
    )


def train_distributed(
    agent_cfg: AgentConfig,
    env: VecEnv,
    *,
    warmstart_reset_steps: bool = False,
) -> None:
    agent_cfg = replace(agent_cfg, train=True)

    agent_cfg = auto_adjust_n_steps(
        agent_cfg,
        env,
        max_rollout_gib=12.0,
        safety_factor=2.0,
    )

    os.makedirs(agent_cfg.model_dir, exist_ok=True)
    save_path = os.path.join(agent_cfg.model_dir, agent_cfg.model_name)
    metrics_csv_path = os.path.join(
        agent_cfg.model_dir,
        f"{os.path.splitext(agent_cfg.model_name)[0]}_metrics.csv",
    )
    perf_csv_path = os.path.join(
        agent_cfg.model_dir,
        f"{os.path.splitext(agent_cfg.model_name)[0]}_perf.csv",
    )

    model = build_model(
        agent_cfg,
        env,
        save_path,
        warmstart_reset_steps=warmstart_reset_steps,
    )
    remote_base_env = _find_remote_base_env(env)
    if remote_base_env is not None:
        remote_base_env.set_perf_csv_path(perf_csv_path)

    target = int(agent_cfg.train_steps)
    already = int(getattr(model, "num_timesteps", 0))
    remaining = max(0, target - already)

    print(f"[TRAIN] alvo_total={target}, ja_treinado={already}, restante={remaining}")

    chunk_max = 50_000
    callback = DebugCallback(log_every=1_000, metrics_csv_path=metrics_csv_path)
    print(f"[TRAIN] MÃ©tricas RL em CSV: {metrics_csv_path}")
    print(f"[TRAIN] MÃ©tricas de estabilidade em CSV: {perf_csv_path}")

    while remaining > 0:
        cur = min(chunk_max, remaining)
        _set_chunk_lr(model, target_total_steps=target)
        _set_chunk_ent_coef(model, target_total_steps=target)

        print(f"[TRAIN] Iniciando chunk: {cur} steps (Restam: {remaining})")
        model.learn(
            total_timesteps=cur,
            reset_num_timesteps=False,
            callback=callback,
            progress_bar=True,
        )
        model.save(save_path)
        remaining -= cur

    print("[TRAIN] Treino concluído.")


def main() -> None:
    args = parse_args()
    normalize_topology(args)
    log_torch_device_info()

    agent_cfg = load_agent_cfg(args.cfg)
    listener, _ = start_listener(args)

    cfg_stack = getattr(agent_cfg, "stack_frames", None)
    stack = int(cfg_stack) if isinstance(cfg_stack, int) and cfg_stack > 0 else int(args.stack)
    print(f"[TRAIN] stack_frames (frames empilhados) = {stack} (cfg/CLI)")

    actors: List[subprocess.Popen] = []
    try:
        actors, conns = launch_actors_staggered_and_accept(args, cfg_path=args.cfg, listener=listener)
        env = build_vec_env(conns, stack=stack, shm_obs=bool(args.shm_obs))
        train_distributed(
            agent_cfg,
            env,
            warmstart_reset_steps=bool(args.warmstart_reset_steps),
        )
        env.close()

    except KeyboardInterrupt:
        print("\n[TRAIN] Interrompido pelo usuário (Ctrl+C).")
    except Exception as e:
        print(f"\n[TRAIN] ERRO FATAL: {e!r}")
    finally:
        print("[TRAIN] Limpando processos...")
        for p in actors:
            try:
                p.terminate()
            except Exception:
                pass
        try:
            listener.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()

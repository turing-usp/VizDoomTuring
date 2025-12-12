#!/usr/bin/env python3
from __future__ import annotations

import time
import argparse
import collections
import os
import subprocess
import sys
from dataclasses import replace
from multiprocessing.connection import Listener, Connection, wait
from typing import List, Tuple, Any, Set, Dict

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv, VecTransposeImage, VecFrameStack
from stable_baselines3.common.callbacks import BaseCallback

from .client import load_agent_cfg
from .config import AgentConfig
from .policies import resolve_algo, build_sb3


class RemoteDMVecEnv(VecEnv):
    """
    VecEnv robusto que delega reset/step para N atores remotos via IPC.
    CORRIGIDO: Usa 'wait' para leitura assíncrona real, evitando travamentos.
    """

    def __init__(self, conns: List[Connection], obs_space: spaces.Space, action_space: spaces.Space):
        assert len(conns) > 0, "É necessário pelo menos 1 conexão de ator"
        self._conns: List[Connection] = conns
        self._waiting: bool = False
        self.num_actors = len(conns)

        super().__init__(
            num_envs=len(conns),
            observation_space=obs_space,
            action_space=action_space,
        )
        print(f"[TRAIN] RemoteDMVecEnv criado com {self.num_envs} envs (jogadores).")

    def _wait_for_replies(self, pendentes: Set[int], timeout_total: float, context: str) -> Dict[int, Any]:
        """
        Aguarda resposta de um conjunto de índices de atores.
        Usa 'wait' (select) para não bloquear a CPU e ler quem responder primeiro.
        """
        results: Dict[int, Any] = {}
        t0 = time.time()
        last_log = time.time()

        # Mapeia conexão -> índice do ator para saber quem respondeu
        conn_map = {self._conns[i]: i for i in pendentes}

        while pendentes:
            # Checagem de Timeout
            if time.time() - t0 > timeout_total:
                raise TimeoutError(
                    f"[TRAIN] TIMEOUT CRÍTICO ({timeout_total}s) em {context}. "
                    f"Atores que não responderam: {sorted(list(pendentes))}. "
                    "O VizDoom pode ter travado ou a CPU está saturada."
                )

            # Log de heartbeat a cada 5s se estiver demorando muito
            if time.time() - last_log > 5.0:
                print(f"[TRAIN][DEBUG] {context}: Aguardando {len(pendentes)} atores: {list(pendentes)[:5]}...", flush=True)
                last_log = time.time()

            # Pega lista de conexões que ainda faltam responder
            conns_to_wait = [self._conns[i] for i in pendentes]

            # Espera até que pelo menos um tenha dados
            ready_conns = wait(conns_to_wait, timeout=0.1)

            for conn in ready_conns:
                idx = conn_map.get(conn)
                if idx is None or idx not in pendentes:
                    continue

                try:
                    msg = conn.recv()

                    if isinstance(msg, dict) and "error" in msg:
                        raise RuntimeError(f"[TRAIN] Ator {idx} reportou erro: {msg['error']}")

                    results[idx] = msg
                    pendentes.remove(idx)
                except EOFError:
                    raise RuntimeError(f"[TRAIN] Conexão com ator {idx} fechada inesperadamente em {context}.")
                except Exception as e:
                    print(f"[TRAIN][ERROR] Exceção ao ler ator {idx}: {e}")
                    raise

        return results

    def reset(self) -> np.ndarray:
        print(f"[TRAIN] Enviando comando 'reset' para {self.num_envs} atores...", flush=True)
        for idx, conn in enumerate(self._conns):
            try:
                conn.send({"cmd": "reset"})
            except Exception as e:
                print(f"[TRAIN][ERROR] Falha ao enviar reset para ator {idx}: {e}")
                raise

        msgs_dict = self._wait_for_replies(
            pendentes=set(range(self.num_envs)),
            timeout_total=120.0,
            context="reset()",
        )

        obs_list = []
        for idx in range(self.num_envs):
            msg = msgs_dict[idx]
            if not isinstance(msg, dict) or "obs" not in msg:
                raise RuntimeError(f"[TRAIN] Resposta inválida reset ator {idx}: {msg}")
            obs_list.append(msg["obs"])

        obs_batch = np.stack(obs_list, axis=0)
        print(f"[TRAIN] reset() concluído com sucesso. Shape={obs_batch.shape}", flush=True)
        return obs_batch

    def step_async(self, actions):
        actions = np.array(actions).reshape((self.num_envs,))
        for idx, (conn, act) in enumerate(zip(self._conns, actions)):
            try:
                conn.send({"cmd": "step", "action": int(act)})
            except Exception as e:
                print(f"[TRAIN][ERROR] Falha ao enviar step para ator {idx}: {e}")
                raise
        self._waiting = True

    def step_wait(self):
        msgs_dict = self._wait_for_replies(
            pendentes=set(range(self.num_envs)),
            timeout_total=60.0,
            context="step_wait()",
        )

        self._waiting = False

        obs_list, rewards, dones, infos = [], [], [], []

        for idx in range(self.num_envs):
            msg = msgs_dict[idx]

            if not isinstance(msg, dict):
                raise RuntimeError(f"[TRAIN] Msg inválida ator {idx}: {msg}")

            obs = msg["obs"]
            rew = float(msg["reward"])
            done = bool(msg["done"])
            info = dict(msg.get("info", {}))

            if done:
                info["terminal_observation"] = obs
                try:
                    self._conns[idx].send({"cmd": "reset"})
                    reset_result = self._wait_for_replies(
                        pendentes={idx},
                        timeout_total=60.0,
                        context=f"reset_pos_done({idx})",
                    )
                    obs = reset_result[idx]["obs"]
                except Exception as e:
                    raise RuntimeError(f"[TRAIN] Erro no reset automático do ator {idx}: {e}")

            obs_list.append(obs)
            rewards.append(rew)
            dones.append(done)
            infos.append(info)

        return (
            np.stack(obs_list, axis=0),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=bool),
            infos,
        )

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

    # Métodos obrigatórios da classe abstrata VecEnv
    def render(self, mode: str = "human"):
        return None

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs):
        raise NotImplementedError()

    def env_is_wrapped(self, wrapper_class, indices=None) -> List[bool]:
        return [False] * self.num_envs

    def get_attr(self, attr_name: str, indices=None) -> List[Any]:
        return [None] * self.num_envs

    def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
        pass


class DebugCallback(BaseCallback):
    def __init__(self, log_every: int = 1_000, reward_window: int = 10_000, verbose: int = 0):
        super().__init__(verbose)
        self.log_every = log_every
        self.reward_window = reward_window
        self._last = 0
        self._recent_rewards = collections.deque(maxlen=reward_window)
        self._rollout_start_time: float | None = None

    def _on_training_start(self) -> None:
        print("[DEBUG] == TRAINING START ==")

    def _on_rollout_start(self) -> None:
        self._rollout_start_time = time.time()
        print(f"[DEBUG] ==== ROLLOUT START (num_timesteps={self.num_timesteps}) ====")

    def _on_rollout_end(self) -> None:
        now = time.time()
        dt = None
        if self._rollout_start_time is not None:
            dt = now - self._rollout_start_time
        print(
            f"[DEBUG] ==== ROLLOUT END (num_timesteps={self.num_timesteps}) "
            + (f"duracao={dt:.2f}s" if dt is not None else "")
        )

    def _on_training_end(self) -> None:
        print("[DEBUG] == TRAINING END ==")

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", None)
        if rewards is not None:
            for r in np.asarray(rewards).ravel():
                self._recent_rewards.append(float(r))

        if self.num_timesteps - self._last >= self.log_every:
            self._last = self.num_timesteps
            if self._recent_rewards:
                mean_r = float(sum(self._recent_rewards) / len(self._recent_rewards))
            else:
                mean_r = float("nan")
            print(
                f"[DEBUG] num_timesteps={self.num_timesteps}, "
                f"mean_reward_window={mean_r:.3f} (últimos {len(self._recent_rewards)} passos)"
            )
        return True


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
    parser.add_argument("--render-all", action="store_true")

    parser.add_argument("--trainer-host", default="127.0.0.1")
    parser.add_argument("--trainer-port", type=int, default=7000)
    parser.add_argument("--auth-key", default="vizdoom_dm")

    return parser.parse_args()


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


def launch_actors(args: argparse.Namespace, cfg_path: str) -> List[subprocess.Popen]:
    num_matches = args.num_matches
    actors_per_match = args.actors_per_match
    procs: List[subprocess.Popen] = []
    module_name = "framework.distributed_actor"

    spawn_delay_actor = 0.5
    spawn_delay_host = 5.0

    print(f"[TRAIN] Lançando {num_matches} partidas...")

    for match_idx in range(num_matches):
        match_port = args.game_port + match_idx
        print(f"\n[TRAIN] === Partida {match_idx} (Porta {match_port}) ===")

        cmd_host = _build_actor_cmd(args, module_name, cfg_path, match_port, True, actors_per_match)
        print(f"[TRAIN] Lançando HOST...")
        procs.append(subprocess.Popen(cmd_host))

        print(f"[TRAIN] Aguardando {spawn_delay_host}s para Host iniciar...")
        time.sleep(spawn_delay_host)

        for local_idx in range(1, actors_per_match):
            cmd_client = _build_actor_cmd(args, module_name, cfg_path, match_port, False, actors_per_match)
            print(f"[TRAIN] Lançando Cliente {local_idx}...")
            procs.append(subprocess.Popen(cmd_client))
            time.sleep(spawn_delay_actor)

    return procs


def _build_actor_cmd(args, module_name, cfg_path, port, is_host, players):
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
    ]
    if is_host:
        cmd.append("--is-host")
        if args.render_all or args.render_host:
            cmd.append("--render")
    elif args.render_all:
        cmd.append("--render")

    return cmd


def accept_actor_conns(listener: Listener, num_actors: int) -> List[Connection]:
    print(f"[TRAIN] Aguardando {num_actors} conexões (Socket)...")
    conns: List[Connection] = []
    for i in range(num_actors):
        conns.append(listener.accept())
        if (i + 1) % 1 == 0:
            print(f"[TRAIN] {i+1}/{num_actors} conectados.")
    return conns


def fetch_spaces(conns: List[Connection]) -> Tuple[spaces.Space, spaces.Space]:
    print("[TRAIN] Solicitando spaces...")
    conns[0].send({"cmd": "get_spaces"})
    msg = conns[0].recv()
    return msg["obs_space"], msg["action_space"]


def build_vec_env(conns: List[Connection], stack: int) -> VecEnv:
    obs_space, action_space = fetch_spaces(conns)
    base_env = RemoteDMVecEnv(conns, obs_space, action_space)
    env = VecTransposeImage(base_env)
    env = VecFrameStack(env, n_stack=stack, channels_order="first")
    return env


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

    max_n_steps = int(
        target_bytes / (safety_factor * n_envs * obs_elems * bytes_per_elem)
    )

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


def build_model(agent_cfg: AgentConfig, env: VecEnv, save_path: str):
    algo_cls = resolve_algo(agent_cfg.policy.algo)
    print("[DEBUG] learn_kwargs:", agent_cfg.policy.learn_kwargs)
    if os.path.exists(save_path):
        print(f"[TRAIN] Carregando modelo: {save_path}")
        try:
            return algo_cls.load(save_path, env=env)
        except Exception as e:
            print(f"[TRAIN] Erro ao carregar: {e}. Criando novo.")

    print(f"[TRAIN] Criando novo modelo ({agent_cfg.policy.algo}).")
    return build_sb3(
        algo_cls,
        "CnnPolicy",
        env,
        agent_cfg.policy.policy_kwargs,
        agent_cfg.policy.learn_kwargs,
    )


# ----------------------------------------------------------------------
# Scheduler de LR por chunk, baseado em train_steps e range (max,min)
# ----------------------------------------------------------------------


def _set_chunk_lr(model, target_total_steps: int) -> None:
    """
    Ajusta learning rate CONSTANTE para o próximo chunk.

    Regras:
    - Usa model._lr_range = (max_lr, min_lr), configurado em build_sb3().
    - progress = num_timesteps / target_total_steps, clamped em [0, 1].
    - LR decai linearmente de max_lr (progress=0) até min_lr (progress=1).
    - Atualiza:
        - model.lr_schedule (função constante)
        - policy.optimizer.param_groups[*]['lr']
        - model.learning_rate (se possível)
    """
    lr_range = getattr(model, "_lr_range", None)
    if not lr_range:
        return

    max_lr, min_lr = lr_range
    total = max(int(target_total_steps), 1)
    t = int(getattr(model, "num_timesteps", 0))
    progress = max(0.0, min(1.0, t / total))

    lr = float(max_lr) - (float(max_lr) - float(min_lr)) * progress

    # sobrescreve o scheduler interno do SB3 para ser constante neste chunk
    def _const_lr(_progress_remaining: float) -> float:
        return lr

    if hasattr(model, "lr_schedule"):
        model.lr_schedule = _const_lr

    # atualiza o otimizador da policy
    opt = getattr(getattr(model, "policy", None), "optimizer", None)
    if opt is not None:
        for pg in opt.param_groups:
            pg["lr"] = lr

    # tenta manter atributo learning_rate coerente (não é obrigatório, mas ajuda logs)
    try:
        model.learning_rate = lr
    except Exception:
        pass

    print(
        f"[TRAIN][LR] num_timesteps={t}/{total}, "
        f"progress={progress:.4f}, lr_chunk={lr}",
        flush=True,
    )


def train_distributed(agent_cfg: AgentConfig, env: VecEnv) -> None:
    """
    Treinador distribuído de um único modelo (compartilhado entre vários atores).

    Comportamento de resume:
    - agent_cfg.train_steps é o alvo total de timesteps.
    - Se existir checkpoint, usamos model.num_timesteps para saber quanto
      já foi treinado e só completamos o restante.

    Scheduler de LR:
    - Se o modelo tiver _lr_range configurado (learning_rate_max/min no YAML),
      aplicamos um LR constante por chunk com decaimento linear em função
      de train_steps.
    """
    agent_cfg = replace(agent_cfg, train=True)

    agent_cfg = auto_adjust_n_steps(
        agent_cfg,
        env,
        max_rollout_gib=4.0,
        safety_factor=4.0,
    )

    os.makedirs(agent_cfg.model_dir, exist_ok=True)
    save_path = os.path.join(agent_cfg.model_dir, agent_cfg.model_name)

    model = build_model(agent_cfg, env, save_path)

    target = int(agent_cfg.train_steps)
    already = int(getattr(model, "num_timesteps", 0))
    remaining = max(0, target - already)

    print(
        f"[TRAIN] alvo_total={target}, "
        f"ja_treinado={already}, restante={remaining}"
    )

    chunk_max = 50_000
    callback = DebugCallback(log_every=1_000)

    while remaining > 0:
        cur = min(chunk_max, remaining)

        # define LR para ESTE chunk, baseado em progresso global
        _set_chunk_lr(model, target_total_steps=target)

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
    agent_cfg = load_agent_cfg(args.cfg)
    listener, _ = start_listener(args)
    stack = getattr(agent_cfg, "stack_frames", None)
    print(f"[TRAIN] stack_frames (frames empilhados) = {stack} (YAML/CLI)")
    actors = []
    try:
        actors = launch_actors(args, cfg_path=args.cfg)
        conns = accept_actor_conns(listener, args.num_actors)
        env = build_vec_env(conns, stack=stack)
        train_distributed(agent_cfg, env)
        env.close()

    except KeyboardInterrupt:
        print("\n[TRAIN] Interrompido pelo usuário (Ctrl+C).")
    except Exception as e:
        print(f"\n[TRAIN] ERRO FATAL: {e}")
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

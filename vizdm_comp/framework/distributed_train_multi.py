#!/usr/bin/env python3
from __future__ import annotations

"""
Treinador distribuído multi-modelo para VizDoom DM.
Versão Corrigida para Windows (Evita Race Condition e Contacting Host loop).
"""

import argparse
import collections
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from multiprocessing.connection import Listener, Connection
from typing import Any, Dict, List, Tuple, Optional

from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv, VecTransposeImage, VecFrameStack

# Importações locais (ajuste conforme a estrutura do seu projeto)
from .client import load_agent_cfg
from .config import AgentConfig
from .distributed_train import (
    RemoteDMVecEnv,
    auto_adjust_n_steps,
    build_model,
    DebugCallback,
)


# ======================================================================
# Parsing dos agentes (YAML:COUNT)
# ======================================================================

@dataclass(frozen=True)
class AgentGroupSpec:
    cfg_path: str
    count: int

    def __repr__(self) -> str:
        return f"AgentGroupSpec(cfg_path={self.cfg_path!r}, count={self.count})"


def parse_agent_spec(spec: str) -> AgentGroupSpec:
    parts = spec.split(":")
    if len(parts) != 2:
        raise ValueError(
            f"Formato inválido para --agent '{spec}'. "
            f"Use: caminho.yaml:count (ex.: example_agent.yaml:3)"
        )
    cfg_path = parts[0].strip()
    try:
        count = int(parts[1])
    except ValueError:
        raise ValueError(f"COUNT inválido em --agent '{spec}' (esperado int).")
    if count <= 0:
        raise ValueError(f"COUNT deve ser > 0 em --agent '{spec}'.")
    return AgentGroupSpec(cfg_path=cfg_path, count=count)


# ======================================================================
# CLI principal
# ======================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Treinador distribuído multi-modelo VizDoom DM"
    )

    parser.add_argument(
        "--agent",
        action="append",
        required=True,
        help="Especificação de agente: caminho_yaml:count (pode repetir).",
    )

    parser.add_argument("--shm-obs", action="store_true", help="Usa shared memory (via Env Var)")
    parser.add_argument("--game-config", default=None, help="Caminho do cfg do jogo")

    parser.add_argument("--num-matches", type=int, default=1, help="Número de partidas em paralelo.")
    parser.add_argument("--game-port", type=int, default=5029, help="Porta base do servidor VizDoom.")
    parser.add_argument("--game-ip", default="127.0.0.1", help="IP do host VizDoom.")
    parser.add_argument("--timelimit", type=float, default=0.0, help="Duração da partida em minutos.")
    parser.add_argument("--stack", type=int, default=4, help="Frames empilhados na entrada da REDE.")
    parser.add_argument("--render-host", action="store_true", help="Renderiza apenas o host.")
    parser.add_argument("--render-all", action="store_true", help="Renderiza todos os atores.")
    parser.add_argument("--trainer-host", default="127.0.0.1", help="Endereço de bind do Listener IPC.")
    parser.add_argument("--trainer-port", type=int, default=7000, help="Porta do Listener IPC.")
    parser.add_argument("--auth-key", default="vizdoom_dm", help="Chave IPC treinador<->atores.")
    parser.add_argument("--chunk-steps", type=int, default=50_000, help="Steps por chunk de treino.")

    parser.add_argument("--map", default="map01", help="Nome do mapa.")
    parser.add_argument("--wad", default=None, help="WAD/PK3.")

    return parser.parse_args()


# ======================================================================
# Infra de listener / atores
# ======================================================================

def start_listener(args: argparse.Namespace, backlog: int) -> Tuple[Listener, Tuple[str, int]]:
    address = (args.trainer_host, args.trainer_port)
    # Aumenta drasticamente o backlog para aguentar várias conexões simultâneas
    safe_backlog = max(64, backlog * 4) 
    
    print(f"[MM-TRAIN] Iniciando Listener IPC em {address} (backlog={safe_backlog})...")
    listener = Listener(
        address,
        backlog=safe_backlog,
        authkey=args.auth_key.encode("utf-8"),
    )
    return listener, address


def _build_actor_cmd_single(
    cfg_path: str,
    players_per_match: int,
    game_ip: str,
    match_port: int,
    timelimit: float,
    trainer_host: str,
    trainer_port: int,
    auth_key: str,
    is_host: bool,
    render_mode: str,
    map_name: str,
    wad: Optional[str],
    game_config: Optional[str] = None,
) -> List[str]:
    cmd = [
        sys.executable,
        "-m",
        "vizdm_comp.framework.distributed_actor",
        "--cfg", cfg_path,
        "--players", str(players_per_match),
        "--port", str(match_port),
        "--join-ip", game_ip,
        "--timelimit", str(timelimit),
        "--trainer-host", trainer_host,
        "--trainer-port", str(trainer_port),
        "--auth-key", auth_key,
        "--map", str(map_name),
    ]
    
    if game_config:
        cmd += ["--game-config", str(game_config)]
        
    if wad:
        cmd += ["--wad", str(wad)]

    if is_host:
        cmd.append("--is-host")
        if render_mode in ("host", "all"):
            cmd.append("--render")
    else:
        if render_mode == "all":
            cmd.append("--render")

    return cmd


def launch_multi_model_actors(
    args: argparse.Namespace,
    agent_specs: List[AgentGroupSpec],
) -> Tuple[List[subprocess.Popen], List[AgentGroupSpec]]:
    procs: List[subprocess.Popen] = []
    per_actor_group_hint: List[AgentGroupSpec] = []

    players_per_match = sum(spec.count for spec in agent_specs)
    print(f"[MM-TRAIN] Jogadores por partida: {players_per_match}")

    render_mode = "all" if args.render_all else ("host" if args.render_host else "none")

    for match_idx in range(args.num_matches):
        # Separa as portas por 10 (5029, 5039, etc)
        match_port = args.game_port + (match_idx * 10)
        print(f"\n[MM-TRAIN] === Partida {match_idx} (porta {match_port}) ===")

        # 1) HOST
        host_spec = agent_specs[0]
        print(f"[MM-TRAIN] Lançando HOST com {host_spec.cfg_path} ...")
        host_cmd = _build_actor_cmd_single(
            cfg_path=host_spec.cfg_path,
            players_per_match=players_per_match,
            game_ip=args.game_ip,
            match_port=match_port,
            timelimit=args.timelimit,
            trainer_host=args.trainer_host,
            trainer_port=args.trainer_port,
            auth_key=args.auth_key,
            is_host=True,
            render_mode=render_mode,
            map_name=args.map,
            wad=args.wad,
            game_config=args.game_config
        )
        print("[MM-TRAIN][CMD-HOST]", " ".join(host_cmd))
        p = subprocess.Popen(host_cmd)
        procs.append(p)
        per_actor_group_hint.append(host_spec)

        print("[MM-TRAIN] Host lançado. Aguardando 6 segundos para estabilizar...")
        time.sleep(6.0)

        # 2) CLIENTES
        print("[MM-TRAIN] Lançando CLIENTES ...")
        for spec_idx, spec in enumerate(agent_specs):
            remaining = spec.count
            if spec_idx == 0:
                remaining -= 1 
            if remaining <= 0:
                continue
            
            for local_idx in range(remaining):
                cmd = _build_actor_cmd_single(
                    cfg_path=spec.cfg_path,
                    players_per_match=players_per_match,
                    game_ip=args.game_ip,
                    match_port=match_port,
                    timelimit=args.timelimit,
                    trainer_host=args.trainer_host,
                    trainer_port=args.trainer_port,
                    auth_key=args.auth_key,
                    is_host=False,
                    render_mode=render_mode,
                    map_name=args.map,
                    wad=args.wad,
                    game_config=args.game_config
                )
                
                p_cli = subprocess.Popen(cmd)
                procs.append(p_cli)
                per_actor_group_hint.append(spec)
                time.sleep(1.0) 

        # --- AQUI ESTÁ A MUDANÇA CRÍTICA ---
        if match_idx < args.num_matches - 1:
            print("[MM-TRAIN] >>> PAUSA DE SEGURANÇA: 15s para não travar o Windows... <<<")
            time.sleep(15.0) 
        # -----------------------------------

    return procs, per_actor_group_hint


def accept_actor_conns(listener: Listener, num_actors: int) -> List[Connection]:
    print(f"[MM-TRAIN] Aguardando {num_actors} conexões (Socket)...")
    conns: List[Connection] = []
    # Loop simples. O listener com backlog alto segura a onda.
    for i in range(num_actors):
        conn = listener.accept()
        conns.append(conn)
        print(f"[MM-TRAIN] {i + 1}/{num_actors} conexões aceitas.")
    return conns


def fetch_spaces(conn: Connection) -> Tuple[spaces.Space, spaces.Space]:
    conn.send({"cmd": "get_spaces"})
    msg = conn.recv()
    return msg["obs_space"], msg["action_space"]


# ======================================================================
# Construção de VecEnvs por grupo
# ======================================================================

@dataclass
class GroupRuntime:
    spec: AgentGroupSpec
    agent_cfg: AgentConfig
    conns: List[Connection]
    env: VecEnv
    model: Any
    save_path: str
    callback: DebugCallback


def build_group_runtimes(
    agent_specs: List[AgentGroupSpec],
    conns: List[Connection],
    stack: int,
) -> List[GroupRuntime]:
    num_actors = len(conns)
    print(f"[MM-TRAIN] Construindo grupos para {num_actors} atores...")

    print("[MM-TRAIN] Fazendo reset inicial em todos os atores para leitura de info['name']...")
    actor_names: List[str] = []
    for c in conns:
        c.send({"cmd": "reset"})
    for idx, c in enumerate(conns):
        msg = c.recv()
        if not isinstance(msg, dict) or "obs" not in msg or "info" not in msg:
            raise RuntimeError(f"[MM-TRAIN] Resposta inválida em reset inicial do ator {idx}: {msg}")
        info = msg.get("info", {})
        name = str(info.get("name", f"Actor{idx}"))
        actor_names.append(name)
        print(f"[MM-TRAIN] Ator {idx}: name={name!r}")

    yaml_name_to_cfg: Dict[str, Tuple[AgentGroupSpec, AgentConfig]] = {}

    for spec in agent_specs:
        agent_cfg = load_agent_cfg(spec.cfg_path)
        yaml_name = agent_cfg.name
        yaml_name_to_cfg[yaml_name] = (spec, agent_cfg)

    group_to_indices: Dict[AgentGroupSpec, List[int]] = collections.defaultdict(list)
    for idx, name in enumerate(actor_names):
        if name not in yaml_name_to_cfg:
            # Fallback: se o nome não bater exato, tenta casar pelo início ou avisa
            raise RuntimeError(
                f"[MM-TRAIN] Ator {idx} reportou name={name!r} mas não achei YAML correspondente. "
                f"Verifique 'name' no YAML."
            )
        spec, _ = yaml_name_to_cfg[name]
        group_to_indices[spec].append(idx)

    group_runtimes: List[GroupRuntime] = []

    for spec in agent_specs:
        indices = group_to_indices.get(spec, [])
        if not indices:
            continue

        agent_cfg = load_agent_cfg(spec.cfg_path)
        stack_for_group = getattr(agent_cfg, "stack_frames", None)
        if stack_for_group is None:
            stack_for_group = stack
        
        os.makedirs(agent_cfg.model_dir, exist_ok=True)
        save_path = os.path.join(agent_cfg.model_dir, agent_cfg.model_name)

        group_conns = [conns[i] for i in indices]
        obs_space, action_space = fetch_spaces(group_conns[0])

        base_env = RemoteDMVecEnv(group_conns, obs_space, action_space)
        env: VecEnv = VecTransposeImage(base_env)
        env = VecFrameStack(env, n_stack=stack_for_group, channels_order="first")

        agent_cfg = auto_adjust_n_steps(
            agent_cfg,
            env,
            max_rollout_gib=4.0,
            safety_factor=4.0,
        )

        print(f"[MM-TRAIN][{spec.cfg_path}] Preparando modelo...")
        model = build_model(agent_cfg, env, save_path)
        callback = DebugCallback(log_every=1_000, reward_window=10_000)

        rt = GroupRuntime(
            spec=spec,
            agent_cfg=agent_cfg,
            conns=group_conns,
            env=env,
            model=model,
            save_path=save_path,
            callback=callback,
        )
        group_runtimes.append(rt)

    return group_runtimes


def train_multi_models(groups: List[GroupRuntime], chunk_steps: int) -> None:
    if not groups:
        return

    remaining_per_group: Dict[str, int] = {}
    for rt in groups:
        key = rt.spec.cfg_path
        target = int(rt.agent_cfg.train_steps)
        already = int(getattr(rt.model, "num_timesteps", 0))
        remaining = max(0, target - already)
        remaining_per_group[key] = remaining
        print(f"[MM-TRAIN][{key}] alvo={target}, ja_foi={already}, falta={remaining}")

    while True:
        all_done = True
        for rt in groups:
            key = rt.spec.cfg_path
            remaining = remaining_per_group[key]
            if remaining <= 0:
                continue

            all_done = False
            cur = min(int(chunk_steps), int(remaining))
            print(f"[MM-TRAIN][{key}] Chunk: {cur} steps (restam {remaining})")
            
            rt.model.learn(
                total_timesteps=cur,
                reset_num_timesteps=False,
                callback=rt.callback,
                progress_bar=True,
            )
            rt.model.save(rt.save_path)
            remaining_per_group[key] = remaining - cur

        if all_done:
            break
    print("[MM-TRAIN] Treino concluído.")


def main() -> None:
    args = parse_args()
    agent_specs: List[AgentGroupSpec] = [parse_agent_spec(s) for s in args.agent]

    total_actors = sum(spec.count for spec in agent_specs) * args.num_matches
    print(f"[MM-TRAIN] Total Atores: {total_actors} ({args.num_matches} partidas)")

    # Backlog alto é essencial para num_matches > 1
    listener, _ = start_listener(args, backlog=total_actors)

    actors: List[subprocess.Popen] = []
    conns: List[Connection] = []
    try:
        actors, _ = launch_multi_model_actors(args, agent_specs)
        conns = accept_actor_conns(listener, total_actors)
        
        groups = build_group_runtimes(agent_specs, conns, args.stack)
        train_multi_models(groups, chunk_steps=args.chunk_steps)

        for rt in groups:
            rt.env.close()

    except KeyboardInterrupt:
        print("\n[MM-TRAIN] Interrompido (Ctrl+C).")
    except Exception as e:
        print(f"\n[MM-TRAIN] ERRO FATAL: {e!r}")
        import traceback
        traceback.print_exc()
    finally:
        print("[MM-TRAIN] Limpando processos...")
        for p in actors:
            p.terminate()
        for c in conns:
            c.close()
        listener.close()

if __name__ == "__main__":
    main()
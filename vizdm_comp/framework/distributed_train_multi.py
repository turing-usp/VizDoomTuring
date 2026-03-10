#!/usr/bin/env python3
from __future__ import annotations
from stable_baselines3.common.vec_env import VecTransposeImage, VecFrameStack
import argparse
import collections
import os
import subprocess
import sys
import time
import threading  # <--- A SOLUÇÃO MÁGICA
from dataclasses import dataclass
from multiprocessing.connection import Listener, Connection
from typing import Any, Dict, List, Tuple, Optional


# Importações locais
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

def parse_agent_spec(spec: str) -> AgentGroupSpec:
    parts = spec.split(":")
    if len(parts) != 2:
        raise ValueError(f"Formato inválido: {spec}")
    return AgentGroupSpec(cfg_path=parts[0].strip(), count=int(parts[1]))

# ======================================================================
# CLI principal
# ======================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", action="append", required=True)
    parser.add_argument("--shm-obs", action="store_true")
    parser.add_argument("--game-config", default=None)
    parser.add_argument("--num-matches", type=int, default=1)
    parser.add_argument("--game-port", type=int, default=5029)
    parser.add_argument("--game-ip", default="127.0.0.1")
    parser.add_argument("--timelimit", type=float, default=0.0)
    parser.add_argument("--stack", type=int, default=4)
    parser.add_argument("--render-host", action="store_true")
    parser.add_argument("--render-all", action="store_true")
    parser.add_argument("--trainer-host", default="127.0.0.1")
    parser.add_argument("--trainer-port", type=int, default=7000)
    parser.add_argument("--auth-key", default="vizdoom_dm")
    parser.add_argument("--chunk-steps", type=int, default=50_000)
    parser.add_argument("--map", default="map01")
    parser.add_argument("--wad", type=str, default="doom2.wad", help="WAD file")
    return parser.parse_args()

# ======================================================================
# Infra de listener / atores
# ======================================================================
def start_listener(args: argparse.Namespace, backlog: int) -> Tuple[Listener, Tuple[str, int]]:
    address = (args.trainer_host, args.trainer_port)
    safe_backlog = max(64, backlog * 4) 
    print(f"[MM-TRAIN] Iniciando Listener IPC em {address} (backlog={safe_backlog})...")
    listener = Listener(address, backlog=safe_backlog, authkey=args.auth_key.encode("utf-8"))
    return listener, address

def _build_actor_cmd_single(cfg_path, players_per_match, game_ip, match_port, timelimit, trainer_host, trainer_port, auth_key, is_host, render_mode, map_name, wad, game_config=None):
    cmd = [
        sys.executable, "-m", "vizdm_comp.framework.distributed_actor",
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
    if game_config: cmd += ["--game-config", str(game_config)]
    if wad: cmd += ["--wad", str(wad)]
    if is_host:
        cmd.append("--is-host")
        if render_mode in ("host", "all"): cmd.append("--render")
    else:
        if render_mode == "all": cmd.append("--render")
    return cmd

def launch_multi_model_actors(args, agent_specs):
    procs = []
    players_per_match = sum(spec.count for spec in agent_specs)
    render_mode = "all" if args.render_all else ("host" if args.render_host else "none")

    for match_idx in range(args.num_matches):
        match_port = args.game_port + (match_idx * 10) # Porta segura
        print(f"\n[MM-TRAIN] === Partida {match_idx} (porta {match_port}) ===")

        # HOST
        host_spec = agent_specs[0]
        host_cmd = _build_actor_cmd_single(host_spec.cfg_path, players_per_match, args.game_ip, match_port, args.timelimit, args.trainer_host, args.trainer_port, args.auth_key, True, render_mode, args.map, args.wad, args.game_config)
        procs.append(subprocess.Popen(host_cmd))
        print("[MM-TRAIN] Host lançado. Aguardando 6 segundos...")
        time.sleep(6.0)

        # CLIENTES
        for spec_idx, spec in enumerate(agent_specs):
            remaining = spec.count
            if spec_idx == 0: remaining -= 1 
            if remaining <= 0: continue
            
            for _ in range(remaining):
                cmd = _build_actor_cmd_single(spec.cfg_path, players_per_match, args.game_ip, match_port, args.timelimit, args.trainer_host, args.trainer_port, args.auth_key, False, render_mode, args.map, args.wad, args.game_config)
                procs.append(subprocess.Popen(cmd))
                time.sleep(1.0) 

        if match_idx < args.num_matches - 1:
            print("[MM-TRAIN] >>> PAUSA DE SEGURANÇA: 10s... <<<")
            time.sleep(10.0) 
    return procs

def accept_actor_conns(listener, num_actors):
    print(f"[MM-TRAIN] Aguardando {num_actors} conexões...")
    conns = []
    for i in range(num_actors):
        conns.append(listener.accept())
        print(f"[MM-TRAIN] {i + 1}/{num_actors} conexões aceitas.")
    return conns

def fetch_spaces(conn):
    conn.send({"cmd": "get_spaces"})
    msg = conn.recv()
    return msg["obs_space"], msg["action_space"]

# ======================================================================
# Construção de Runtimes
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

def build_group_runtimes(agent_specs, conns, stack):
    num_actors = len(conns)
    print(f"[MM-TRAIN] Resetando {num_actors} atores para identificação...")
    
    actor_names = []
    for c in conns: c.send({"cmd": "reset"})
    for c in conns:
        msg = c.recv()
        info = msg.get("info", {})
        actor_names.append(str(info.get("name", "Unknown")))

    yaml_name_to_cfg = {}
    for spec in agent_specs:
        cfg = load_agent_cfg(spec.cfg_path)
        yaml_name_to_cfg[cfg.name] = (spec, cfg)

    group_to_indices = collections.defaultdict(list)
    for idx, name in enumerate(actor_names):
        if name not in yaml_name_to_cfg:
             # Tenta fallback pelo config path se o nome não bater
             print(f"[WARN] Nome '{name}' não achado nos YAMLs. Usando ordem de criação.")
             # Lógica simplificada de fallback baseada na ordem
             # (Isso é arriscado, ideal é arrumar os nomes nos YAMLs)
        else:
             spec, _ = yaml_name_to_cfg[name]
             group_to_indices[spec].append(idx)

    group_runtimes = []
    for spec in agent_specs:
        indices = group_to_indices.get(spec, [])
        if not indices: continue

        agent_cfg = load_agent_cfg(spec.cfg_path)
        stack_val = getattr(agent_cfg, "stack_frames", stack)
        
        os.makedirs(agent_cfg.model_dir, exist_ok=True)
        save_path = os.path.join(agent_cfg.model_dir, agent_cfg.model_name)
        group_conns = [conns[i] for i in indices]
        
        obs_space, action_space = fetch_spaces(group_conns[0])
        base_env = RemoteDMVecEnv(group_conns, obs_space, action_space)
        env = VecTransposeImage(base_env)
        env = VecFrameStack(env, n_stack=stack_val, channels_order="first")
        
        agent_cfg = auto_adjust_n_steps(agent_cfg, env)
        print(f"[MM-TRAIN][{spec.cfg_path}] Preparando modelo...")
        model = build_model(agent_cfg, env, save_path)
        callback = DebugCallback(log_every=1_000, reward_window=10_000)

        group_runtimes.append(GroupRuntime(spec, agent_cfg, group_conns, env, model, save_path, callback))

    return group_runtimes

# ======================================================================
# Loop principal com THREADING (A Mágica)
# ======================================================================
def train_chunk_threaded(rt: GroupRuntime, steps: int):
    """Função executada em thread separada para cada modelo."""
    try:
        rt.model.learn(
            total_timesteps=steps,
            reset_num_timesteps=False,
            callback=rt.callback,
            progress_bar=False,
        )
        rt.model.save(rt.save_path)
    except Exception as e:
        print(f"[THREAD-ERROR] Falha no treino de {rt.spec.cfg_path}: {e}")

def train_multi_models(groups: List[GroupRuntime], chunk_steps: int):
    if not groups: return

    # Mapa de controle de quanto falta treinar
    remaining_per_group = {}
    for rt in groups:
        key = rt.spec.cfg_path
        target = int(rt.agent_cfg.train_steps)
        already = int(getattr(rt.model, "num_timesteps", 0))
        remaining = max(0, target - already)
        remaining_per_group[key] = remaining
        print(f"[MM-TRAIN][{key}] Alvo={target}, Já foi={already}, Falta={remaining}")

    while True:
        # Verifica se todos acabaram
        active_groups = [rt for rt in groups if remaining_per_group[rt.spec.cfg_path] > 0]
        if not active_groups:
            break

        threads = []
        print(f"\n[MM-TRAIN] Iniciando rodada de treino PARALELO para {len(active_groups)} grupos...")
        
        for rt in active_groups:
            key = rt.spec.cfg_path
            remaining = remaining_per_group[key]
            cur = min(int(chunk_steps), int(remaining))
            
            print(f"[MM-TRAIN] -> Thread: {key} (Steps: {cur})")
            
            # Cria a thread para rodar o .learn() deste agente
            t = threading.Thread(target=train_chunk_threaded, args=(rt, cur))
            threads.append((t, rt, cur))
            t.start()

        # Espera TODOS terminarem. 
        # Como o GIL do Python libera em I/O (sockets), e o VizDoom é I/O bound aqui, 
        # eles vão conseguir trocar dados simultaneamente!
        for t, rt, cur in threads:
            t.join()
            # Atualiza contagem
            remaining_per_group[rt.spec.cfg_path] -= cur

    print("[MM-TRAIN] Treino multi-modelo concluído.")

# ======================================================================
# Main
# ======================================================================
def main():
    args = parse_args()
    agent_specs = [parse_agent_spec(s) for s in args.agent]
    total_actors = sum(spec.count for spec in agent_specs) * args.num_matches
    
    print(f"[MM-TRAIN] {args.num_matches} partidas, {total_actors} atores total.")
    listener, _ = start_listener(args, backlog=total_actors)

    actors = []
    conns = []
    try:
        actors = launch_multi_model_actors(args, agent_specs)
        conns = accept_actor_conns(listener, total_actors)
        groups = build_group_runtimes(agent_specs, conns, args.stack)
        
        # AQUI CHAMAMOS A NOVA VERSÃO COM THREADS
        train_multi_models(groups, chunk_steps=args.chunk_steps)

        for rt in groups: rt.env.close()

    except KeyboardInterrupt:
        print("\n[MM-TRAIN] Interrompido.")
    except Exception as e:
        print(f"\n[MM-TRAIN] ERRO: {e!r}")
        import traceback
        traceback.print_exc()
    finally:
        print("[MM-TRAIN] Limpando...")
        for p in actors: p.terminate()
        for c in conns: c.close()
        listener.close()

if __name__ == "__main__":
    main()
    # --- A VACINA ANTI-DEADLOCK DO LINUX ---
    import multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
        print("[INIT] Forçando multiprocessing para 'spawn' (Compatibilidade Linux/C++)")
    except RuntimeError:
        pass
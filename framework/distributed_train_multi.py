#!/usr/bin/env python3
from __future__ import annotations

"""
Treinador distribuído multi-modelo para VizDoom DM.

Ideia:
- Você passa vários agentes via CLI: --agent yaml1:count1 --agent yaml2:count2 ...
- Cada grupo (yaml, count) gera `count` atores remotos usando aquele YAML.
- Todos conectam ao mesmo servidor VizDoom (deathmatch compartilhado).
- No trainer, cada grupo de atores é envolvido em um RemoteDMVecEnv próprio
  e recebe um modelo SB3 próprio (PPO / A2C / DQN), usando a PolicyConfig
  lida do respectivo YAML.

Limitação proposital:
- Cada modelo enxerga apenas os seus próprios atores (RemoteDMVecEnv isolado).
- O jogo é compartilhado via UDP pela lógica de host/clients do VizDoom.
- Isso permite multi-modelo sem reescrever o core do PPO do SB3.

Uso exemplo:

    python -m vizdm_comp.framework.distributed_train_multi \
        --agent vizdm_comp/example_agent.yaml:4 \
        --agent bob.yaml:4 \
        --num-matches 1 \
        --game-port 5029 \
        --stack 4
"""

import argparse
import collections
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from multiprocessing.connection import Listener, Connection
from typing import Any, Dict, List, Tuple

from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv, VecTransposeImage, VecFrameStack

from .client import load_agent_cfg
from .config import AgentConfig
from .distributed_train import (
    RemoteDMVecEnv,          # VecEnv remoto já testado
    auto_adjust_n_steps,     # ajuste automático de n_steps por memória
    build_model,             # criação/carregamento de modelo SB3
    DebugCallback,           # logging de rewards / timesteps
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
    """
    Formato aceito: "caminho.yaml:count"
    Ex: "example_agent.yaml:3"
    """
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
        help="Especificação de agente: caminho_yaml:count (pode repetir). "
             "Ex: --agent example_agent.yaml:4 --agent bob.yaml:4",
    )

    parser.add_argument(
        "--num-matches",
        type=int,
        default=1,
        help="Número de partidas em paralelo (default: 1).",
    )

    parser.add_argument(
        "--game-port",
        type=int,
        default=5029,
        help="Porta base do servidor VizDoom (default: 5029). "
             "Cada partida usa game-port + idx.",
    )

    parser.add_argument(
        "--game-ip",
        default="127.0.0.1",
        help="IP do host VizDoom (default: 127.0.0.1).",
    )

    parser.add_argument(
        "--timelimit",
        type=float,
        default=0.0,
        help="Duração da partida em minutos (0 = infinito).",
    )

    parser.add_argument(
        "--stack",
        type=int,
        default=4,
        help="Frames empilhados na entrada da REDE (default: 4).",
    )

    parser.add_argument(
        "--render-host",
        action="store_true",
        help="Renderiza apenas o host (por partida).",
    )

    parser.add_argument(
        "--render-all",
        action="store_true",
        help="Renderiza todos os atores (cuidado com performance).",
    )

    parser.add_argument(
        "--trainer-host",
        default="127.0.0.1",
        help="Endereço de bind do Listener IPC treinador<->atores.",
    )

    parser.add_argument(
        "--trainer-port",
        type=int,
        default=7000,
        help="Porta do Listener IPC treinador<->atores.",
    )

    parser.add_argument(
        "--auth-key",
        default="vizdoom_dm",
        help="Chave IPC treinador<->atores.",
    )

    parser.add_argument(
        "--chunk-steps",
        type=int,
        default=50_000,
        help="Quantidade de steps por chunk de treino para cada modelo.",
    )

    return parser.parse_args()


# ======================================================================
# Infra de listener / atores
# ======================================================================

def start_listener(args: argparse.Namespace, backlog: int) -> Tuple[Listener, Tuple[str, int]]:
    address = (args.trainer_host, args.trainer_port)
    print(f"[MM-TRAIN] Iniciando Listener IPC em {address} (backlog={backlog})...")
    listener = Listener(
        address,
        backlog=max(1, backlog),
        authkey=args.auth_key.encode("utf-8"),
    )
    return listener, address


def _build_actor_cmd_single(
    module_name: str,
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
) -> List[str]:
    """
    Monta comando para lançar 1 ator remoto do módulo distributed_actor
    com o YAML específico. O próprio YAML define train/train_steps/policy/etc.
    """
    cmd = [
        sys.executable,
        "-m",
        "vizdm_comp.framework.distributed_actor",
        "--cfg",
        cfg_path,
        "--players",
        str(players_per_match),
        "--port",
        str(match_port),
        "--join-ip",
        game_ip,
        "--timelimit",
        str(timelimit),
        "--trainer-host",
        trainer_host,
        "--trainer-port",
        str(trainer_port),
        "--auth-key",
        auth_key,
    ]
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
    """
    Lança atores para todas as partidas e todos os grupos de agentes.

    Retorna:
        - lista de processos Popen (um por ator).
        - lista flatten de AgentGroupSpec repetidos conforme os atores (um por ator).
          (apenas para debug; o agrupamento real é feito por YAML depois).
    """
    procs: List[subprocess.Popen] = []
    per_actor_group_hint: List[AgentGroupSpec] = []

    players_per_match = sum(spec.count for spec in agent_specs)
    print(f"[MM-TRAIN] Jogadores por partida: {players_per_match}")

    render_mode = "all" if args.render_all else ("host" if args.render_host else "none")

    for match_idx in range(args.num_matches):
        match_port = args.game_port + match_idx
        print(f"\n[MM-TRAIN] === Partida {match_idx} (porta {match_port}) ===")

        # 1) HOST: primeiro agente da lista
        host_spec = agent_specs[0]
        print(f"[MM-TRAIN] Lançando HOST com {host_spec.cfg_path} ...")
        host_cmd = _build_actor_cmd_single(
            module_name="vizdm_comp.framework.distributed_actor",
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
        )
        print("[MM-TRAIN][CMD-HOST]", " ".join(host_cmd))
        p = subprocess.Popen(host_cmd)
        procs.append(p)
        per_actor_group_hint.append(host_spec)

        # Delay para Host criar a sala UDP
        time.sleep(5.0)

        # 2) CLIENTES: todos os specs, respeitando counts, mas já usamos 1 do host_spec
        print("[MM-TRAIN] Lançando CLIENTES ...")
        for spec_idx, spec in enumerate(agent_specs):
            remaining = spec.count
            if spec_idx == 0:
                remaining -= 1
            if remaining <= 0:
                continue
            for local_idx in range(remaining):
                cmd = _build_actor_cmd_single(
                    module_name="vizdm_comp.framework.distributed_actor",
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
                )
                print(
                    f"[MM-TRAIN]   Cliente match={match_idx}, "
                    f"spec={spec.cfg_path}, idx_local={local_idx}"
                )
                print("[MM-TRAIN][CMD-CLI]", " ".join(cmd))
                p_cli = subprocess.Popen(cmd)
                procs.append(p_cli)
                per_actor_group_hint.append(spec)
                time.sleep(0.5)  # evita avalanche de processos

    return procs, per_actor_group_hint


def accept_actor_conns(
    listener: Listener,
    num_actors: int,
) -> List[Connection]:
    print(f"[MM-TRAIN] Aguardando {num_actors} conexões (Socket)...")
    conns: List[Connection] = []
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
    env: VecEnv              # VecFrameStack(VecTransposeImage(RemoteDMVecEnv))
    model: Any
    save_path: str
    callback: DebugCallback

def build_group_runtimes(
    agent_specs: List[AgentGroupSpec],
    conns: List[Connection],
    stack: int,
) -> List[GroupRuntime]:
    """
    Constrói, para cada grupo de agentes (YAML), um RemoteDMVecEnv separado com
    as conexões correspondentes e instancia o modelo SB3 apropriado.

    Estratégia de agrupamento:
    - Após aceitar as conexões, fazemos um reset inicial em TODOS os atores
      para descobrir o 'name' de cada env (AgentConfig.name do YAML).
    - Em seguida, mapeamos cada conexão para o YAML cujo AgentConfig.name coincidir.
      (supõe-se que cada YAML tenha um 'name' distinto).

    O número de frames empilhados (stack) é definido por agente:
    - Se AgentConfig tiver 'stack_frames', usa esse valor.
    - Caso contrário, usa o 'stack' passado como argumento (CLI).
    """
    num_actors = len(conns)
    print(f"[MM-TRAIN] Construindo grupos para {num_actors} atores...")

    # 1) Reset inicial em todos os atores para descobrir 'name'
    print("[MM-TRAIN] Fazendo reset inicial em todos os atores para leitura de info['name']...")
    actor_names: List[str] = []
    for idx, c in enumerate(conns):
        c.send({"cmd": "reset"})
    for idx, c in enumerate(conns):
        msg = c.recv()
        if not isinstance(msg, dict) or "obs" not in msg or "info" not in msg:
            raise RuntimeError(f"[MM-TRAIN] Resposta inválida em reset inicial do ator {idx}: {msg}")
        info = msg.get("info", {})
        name = str(info.get("name", f"Actor{idx}"))
        actor_names.append(name)
        print(f"[MM-TRAIN] Ator {idx}: name={name!r}")

    # 2) Carrega AgentConfig de cada YAML e prepara mapa name->cfg/spec
    yaml_name_to_cfg: Dict[str, Tuple[AgentGroupSpec, AgentConfig]] = {}

    for spec in agent_specs:
        agent_cfg = load_agent_cfg(spec.cfg_path)
        yaml_name = agent_cfg.name
        if yaml_name in yaml_name_to_cfg:
            print(
                f"[MM-TRAIN][WARN] Nome de agente duplicado entre YAMLs: {yaml_name!r}. "
                "Certifique-se de usar 'name' diferente em cada YAML."
            )
        yaml_name_to_cfg[yaml_name] = (spec, agent_cfg)
        print(f"[MM-TRAIN] YAML {spec.cfg_path} -> agent.name={yaml_name!r}")

    # 3) Agrupa índices de conexões por YAML, usando o 'name'
    group_to_indices: Dict[AgentGroupSpec, List[int]] = collections.defaultdict(list)
    for idx, name in enumerate(actor_names):
        if name not in yaml_name_to_cfg:
            raise RuntimeError(
                f"[MM-TRAIN] Ator {idx} reportou name={name!r} sem YAML correspondente. "
                f"Ajuste o campo 'name' no YAML."
            )
        spec, _ = yaml_name_to_cfg[name]
        group_to_indices[spec].append(idx)

    for spec in agent_specs:
        print(
            f"[MM-TRAIN] Grupo {spec.cfg_path}: atores índices "
            f"{group_to_indices.get(spec, [])}"
        )

    # 4) Para cada grupo, monta VecEnv, ajusta n_steps e carrega/cria modelo
    group_runtimes: List[GroupRuntime] = []

    for spec in agent_specs:
        indices = group_to_indices.get(spec, [])
        if not indices:
            continue

        # Carrega AgentConfig (com reward, render_settings, policy, etc.)
        agent_cfg = load_agent_cfg(spec.cfg_path)

        # Decide stack deste grupo: YAML > CLI default
        stack_for_group = getattr(agent_cfg, "stack_frames", None)
        if stack_for_group is None:
            stack_for_group = stack
        print(
            f"[MM-TRAIN][{spec.cfg_path}] stack_frames (frames empilhados) = "
            f"{stack_for_group} (YAML/CLI)"
        )

        # Garante diretório do modelo
        os.makedirs(agent_cfg.model_dir, exist_ok=True)
        save_path = os.path.join(agent_cfg.model_dir, agent_cfg.model_name)

        # Sublista de conexões deste grupo
        group_conns = [conns[i] for i in indices]

        # Descobre spaces usando a primeira conexão
        obs_space, action_space = fetch_spaces(group_conns[0])

        # Base VecEnv remoto
        base_env = RemoteDMVecEnv(group_conns, obs_space, action_space)

        # Wrappers de imagem + stack (por modelo)
        env: VecEnv = VecTransposeImage(base_env)
        env = VecFrameStack(env, n_stack=stack_for_group, channels_order="first")

        # Ajuste automático de n_steps com base na memória
        agent_cfg = auto_adjust_n_steps(
            agent_cfg,
            env,
            max_rollout_gib=4.0,
            safety_factor=4.0,
        )

        # Cria ou carrega modelo SB3
        print(f"[MM-TRAIN][{spec.cfg_path}] Preparando modelo...")
        model = build_model(agent_cfg, env, save_path)

        # Callback de debug por grupo
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

# ======================================================================
# Loop principal multi-modelo
# ======================================================================

def train_multi_models(
    groups: List[GroupRuntime],
    chunk_steps: int,
) -> None:
    """
    Treina vários modelos (um por YAML/grupo) em round-robin.

    Comportamento de resume:
    - Para cada grupo (YAML), interpretamos agent_cfg.train_steps como
      "total de timesteps desejados".
    - Lemos model.num_timesteps do checkpoint carregado.
    - Treinamos apenas até atingir esse alvo total.
    - Se você interromper (Ctrl+C) e rodar de novo com o mesmo YAML
      e o mesmo caminho de modelo, o treino continua de onde parou.
    """
    if not groups:
        print("[MM-TRAIN] Nenhum grupo para treinar.")
        return

    # Quanto cada modelo ainda deve treinar (pode ser diferente entre YAMLs)
    remaining_per_group: Dict[str, int] = {}

    for rt in groups:
        key = rt.spec.cfg_path
        target = int(rt.agent_cfg.train_steps)
        already = int(getattr(rt.model, "num_timesteps", 0))
        remaining = max(0, target - already)

        remaining_per_group[key] = remaining

        print(
            f"[MM-TRAIN][{key}] alvo_total={target}, "
            f"ja_treinado={already}, restante={remaining}"
        )

    # Loop até todos os modelos encerrarem seu treino
    while True:
        all_done = True

        for rt in groups:
            key = rt.spec.cfg_path
            remaining = remaining_per_group[key]
            if remaining <= 0:
                continue

            all_done = False
            cur = min(chunk_steps, remaining)
            print(
                f"[MM-TRAIN][{key}] Iniciando chunk de treino: {cur} steps "
                f"(restam {remaining})"
            )
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

    print("[MM-TRAIN] Treino multi-modelo concluído.")

# ======================================================================
# main()
# ======================================================================

def main() -> None:
    args = parse_args()

    # Converte specs
    agent_specs: List[AgentGroupSpec] = [parse_agent_spec(s) for s in args.agent]

    for spec in agent_specs:
        print(f"[MM-TRAIN] Grupo: {spec.cfg_path} x {spec.count}")

    total_actors_per_match = sum(spec.count for spec in agent_specs)
    total_actors = total_actors_per_match * args.num_matches
    print(
        f"[MM-TRAIN] Topologia: {args.num_matches} partidas, "
        f"{total_actors_per_match} atores por partida, "
        f"{total_actors} atores no total."
    )

    listener, _ = start_listener(args, backlog=total_actors)

    actors: List[subprocess.Popen] = []
    conns: List[Connection] = []
    try:
        # 1) Lança atores remotos (host + clientes)
        actors, _ = launch_multi_model_actors(args, agent_specs)

        # 2) Aceita conexões TCP (trainer<->atores)
        conns = accept_actor_conns(listener, total_actors)

        # 3) Agrupa conexões por YAML e constroi models/envs
        groups = build_group_runtimes(
            agent_specs=agent_specs,
            conns=conns,
            stack=args.stack,
        )

        # 4) Treino multi-modelo
        train_multi_models(groups, chunk_steps=args.chunk_steps)

        # 5) Fecha envs
        for rt in groups:
            try:
                rt.env.close()
            except Exception:
                pass

    except KeyboardInterrupt:
        print("\n[MM-TRAIN] Interrompido pelo usuário (Ctrl+C).")
    except Exception as e:
        print(f"\n[MM-TRAIN] ERRO FATAL: {e!r}")
    finally:
        print("[MM-TRAIN] Limpando processos e sockets...")
        for p in actors:
            try:
                p.terminate()
            except Exception:
                pass
        for c in conns:
            try:
                c.close()
            except Exception:
                pass
        try:
            listener.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()

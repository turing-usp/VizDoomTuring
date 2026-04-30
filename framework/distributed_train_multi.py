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
        --stack 4 \
        --map map01 \
        --wad mypack.wad
"""

import argparse
import collections
import csv
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from multiprocessing.connection import Listener, Connection
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv, VecFrameStack

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


@dataclass(frozen=True)
class MatchPlanEntry:
    scenario: Optional[str]
    map_name: str
    count: int
    wad: Optional[str] = None


def parse_match_spec(spec: str) -> MatchPlanEntry:
    parts = [part.strip() for part in str(spec).split("|")]
    if len(parts) not in (3, 4):
        raise ValueError(
            f"Formato invalido para --match '{spec}'. "
            "Use: SCENARIO|MAP|COUNT ou SCENARIO|MAP|COUNT|WAD"
        )

    scenario = parts[0] or None
    map_name = parts[1]
    count_str = parts[2]
    wad = parts[3] or None if len(parts) == 4 else None

    if not map_name:
        raise ValueError(f"MAP vazio em --match '{spec}'.")

    try:
        count = int(count_str)
    except ValueError as e:
        raise ValueError(f"COUNT invalido em --match '{spec}' (esperado int).") from e

    if count <= 0:
        raise ValueError(f"COUNT deve ser > 0 em --match '{spec}'.")

    return MatchPlanEntry(
        scenario=scenario,
        map_name=map_name,
        count=count,
        wad=wad,
    )


def expand_match_plan(args: argparse.Namespace) -> List[MatchPlanEntry]:
    if not getattr(args, "match", None):
        return [
            MatchPlanEntry(
                scenario=args.scenario,
                map_name=args.map,
                count=1,
                wad=args.wad,
            )
            for _ in range(int(args.num_matches))
        ]

    plan: List[MatchPlanEntry] = []
    for raw_spec in args.match:
        spec = parse_match_spec(raw_spec)
        for _ in range(spec.count):
            plan.append(
                MatchPlanEntry(
                    scenario=spec.scenario,
                    map_name=spec.map_name,
                    count=1,
                    wad=spec.wad,
                )
            )
    return plan


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

    # --- NEW: map and wad passed to every actor ---
    parser.add_argument(
        "--map",
        default="map01",
        help="Nome do mapa (ex.: map01, MAP01). Será passado a todos os atores.",
    )
    parser.add_argument(
        "--wad",
        default=None,
        help="WAD/PK3 (nome em framework/maps/ OU caminho completo). Será passado a todos os atores.",
    )

    parser.add_argument("--host-start-delay", type=float, default=1.5)
    parser.add_argument("--actor-start-delay", type=float, default=0.05)
    parser.add_argument("--frame-skip", type=int, default=8)
    parser.add_argument("--ticrate", type=int, default=30)
    parser.add_argument("--shm-obs", action="store_true")
    parser.add_argument("--warmstart-reset-steps", action="store_true")
    parser.add_argument("--progress-bar", action="store_true")
    parser.add_argument("--play", action="store_true")
    parser.add_argument("--play-steps", type=int, default=0)
    parser.add_argument("--scenario", default=None)
    parser.add_argument(
        "--match",
        action="append",
        help="Mistura de partidas: SCENARIO|MAP|COUNT ou SCENARIO|MAP|COUNT|WAD. Pode repetir.",
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
    scenario: Optional[str],
    frame_skip: int,
    ticrate: int,
) -> List[str]:
    """
    Monta comando para lançar 1 ator remoto do módulo distributed_actor
    com o YAML específico.

    Além do YAML, passamos parâmetros de mapa/WAD para garantir que todos
    os atores carreguem exatamente o mesmo conteúdo.
    """
    cmd = [
        sys.executable,
        "-m",
        "framework.distributed_actor",
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
        "--map",
        str(map_name),
        "--frame-skip",
        str(frame_skip),
        "--ticrate",
        str(ticrate),
    ]
    if scenario:
        cmd += ["--scenario", str(scenario)]
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
    match_plan = expand_match_plan(args)

    for match_idx, match_entry in enumerate(match_plan):
        match_port = args.game_port + match_idx
        print(f"\n[MM-TRAIN] === Partida {match_idx} (porta {match_port}) ===")
        print(
            f"[MM-TRAIN] Match config: scenario={match_entry.scenario!r} "
            f"map={match_entry.map_name!r} wad={match_entry.wad!r}"
        )

        # 1) HOST: primeiro agente da lista
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
            map_name=match_entry.map_name,
            wad=match_entry.wad,
            scenario=match_entry.scenario,
            frame_skip=args.frame_skip,
            ticrate=args.ticrate,
        )
        print("[MM-TRAIN][CMD-HOST]", " ".join(host_cmd))
        p = subprocess.Popen(host_cmd)
        procs.append(p)
        per_actor_group_hint.append(host_spec)

        # Delay para Host criar a sala UDP
        time.sleep(max(0.0, float(args.host_start_delay)))

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
                    map_name=match_entry.map_name,
                    wad=match_entry.wad,
                    scenario=match_entry.scenario,
                    frame_skip=args.frame_skip,
                    ticrate=args.ticrate,
                )
                print(
                    f"[MM-TRAIN]   Cliente match={match_idx}, "
                    f"spec={spec.cfg_path}, idx_local={local_idx}"
                )
                print("[MM-TRAIN][CMD-CLI]", " ".join(cmd))
                p_cli = subprocess.Popen(cmd)
                procs.append(p_cli)
                per_actor_group_hint.append(spec)
                time.sleep(max(0.0, float(args.actor_start_delay)))

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
    env: VecEnv              # VecFrameStack(RemoteDMVecEnv)
    model: Any
    save_path: str
    callback: DebugCallback


def build_group_runtimes(
    agent_specs: List[AgentGroupSpec],
    conns: List[Connection],
    stack: int,
    *,
    shm_obs: bool = False,
    require_existing_models: bool = False,
    warmstart_reset_steps: bool = False,
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
        if require_existing_models and not os.path.exists(save_path):
            raise FileNotFoundError(f"[MM-PLAY][{spec.cfg_path}] Modelo nao encontrado: {save_path}")

        # Sublista de conexões deste grupo
        group_conns = [conns[i] for i in indices]

        # Descobre spaces usando a primeira conexão
        obs_space, action_space = fetch_spaces(group_conns[0])

        # Base VecEnv remoto
        base_env = RemoteDMVecEnv(group_conns, obs_space, action_space, shm_obs=shm_obs)

        # Wrappers de imagem + stack (por modelo)
        env: VecEnv = base_env
        channels_order: Any = {"image": "first", "state": None} if isinstance(obs_space, spaces.Dict) else "first"
        env = VecFrameStack(env, n_stack=stack_for_group, channels_order=channels_order)

        # Ajuste automático de n_steps com base na memória
        agent_cfg = auto_adjust_n_steps(
            agent_cfg,
            env,
            max_rollout_gib=4.0,
            safety_factor=4.0,
        )

        # Cria ou carrega modelo SB3
        print(f"[MM-TRAIN][{spec.cfg_path}] Preparando modelo...")
        model = build_model(
            agent_cfg,
            env,
            save_path,
            warmstart_reset_steps=warmstart_reset_steps,
        )

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
    *,
    progress_bar: bool = True,
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

    while True:
        all_done = True

        for rt in groups:
            key = rt.spec.cfg_path
            remaining = remaining_per_group[key]
            if remaining <= 0:
                continue

            all_done = False
            cur = min(int(chunk_steps), int(remaining))
            print(
                f"[MM-TRAIN][{key}] Iniciando chunk de treino: {cur} steps "
                f"(restam {remaining})"
            )
            rt.model.learn(
                total_timesteps=cur,
                reset_num_timesteps=False,
                callback=rt.callback,
                progress_bar=progress_bar,
            )
            rt.model.save(rt.save_path)

            remaining_per_group[key] = remaining - cur

        if all_done:
            break

    print("[MM-TRAIN] Treino multi-modelo concluído.")


def _prepare_model_for_play(rt: GroupRuntime) -> None:
    target_steps = max(0, int(getattr(rt.agent_cfg, "train_steps", 0)))
    if target_steps > 0:
        rt.model.num_timesteps = max(int(getattr(rt.model, "num_timesteps", 0)), target_steps)
    if hasattr(rt.model, "_current_progress_remaining"):
        rt.model._current_progress_remaining = 0.0
    if hasattr(rt.model, "exploration_rate"):
        rt.model.exploration_rate = 0.0
    policy = getattr(rt.model, "policy", None)
    if policy is not None and hasattr(policy, "set_training_mode"):
        policy.set_training_mode(False)


def _append_play_results_csv(groups: List[GroupRuntime], results: Dict[str, Dict[str, float]]) -> None:
    if not groups:
        return
    out_dir = groups[0].agent_cfg.model_dir
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "multi_play_results.csv")
    exists = os.path.exists(out_path)
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "time",
                "agent_yaml",
                "model_name",
                "actors",
                "steps",
                "frags",
                "deaths",
                "reward_total",
                "frags_per_actor",
            ],
        )
        if not exists:
            writer.writeheader()
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        for rt in groups:
            row = results[rt.spec.cfg_path]
            actors = max(1, int(row["actors"]))
            writer.writerow(
                {
                    "time": stamp,
                    "agent_yaml": rt.spec.cfg_path,
                    "model_name": rt.agent_cfg.model_name,
                    "actors": actors,
                    "steps": int(row["steps"]),
                    "frags": round(row["frags"], 6),
                    "deaths": round(row["deaths"], 6),
                    "reward_total": round(row["reward_total"], 6),
                    "frags_per_actor": round(row["frags"] / actors, 6),
                }
            )
    print(f"[MM-PLAY] Resultado salvo em CSV: {out_path}", flush=True)


def play_multi_models(groups: List[GroupRuntime], play_steps: int) -> None:
    if not groups:
        print("[MM-PLAY] Nenhum grupo para assistir.")
        return

    observations: Dict[str, Any] = {}
    latest_frags: Dict[str, np.ndarray] = {}
    latest_deaths: Dict[str, np.ndarray] = {}
    reward_totals: Dict[str, float] = {}

    for rt in groups:
        _prepare_model_for_play(rt)
        key = rt.spec.cfg_path
        observations[key] = rt.env.reset()
        latest_frags[key] = np.zeros((rt.env.num_envs,), dtype=np.float32)
        latest_deaths[key] = np.zeros((rt.env.num_envs,), dtype=np.float32)
        reward_totals[key] = 0.0
        print(
            f"[MM-PLAY][{key}] modelo={rt.save_path} actors={rt.env.num_envs} "
            "deterministic=True exploration_rate=0 sem treino/sem save",
            flush=True,
        )

    max_steps = max(0, int(play_steps))
    steps = 0
    try:
        while max_steps <= 0 or steps < max_steps:
            for rt in groups:
                key = rt.spec.cfg_path
                action, _state = rt.model.predict(observations[key], deterministic=True)
                rt.env.step_async(action)

            for rt in groups:
                key = rt.spec.cfg_path
                obs, rewards, _dones, infos = rt.env.step_wait()
                observations[key] = obs
                reward_totals[key] += float(np.sum(rewards))
                for idx, info in enumerate(infos):
                    if "frags" in info:
                        latest_frags[key][idx] = float(info["frags"])
                    if "deaths" in info:
                        latest_deaths[key][idx] = float(info["deaths"])
            steps += 1

            if steps % 1000 == 0:
                parts = []
                for rt in groups:
                    key = rt.spec.cfg_path
                    parts.append(
                        f"{os.path.basename(key)} frags={float(np.sum(latest_frags[key])):.0f} "
                        f"deaths={float(np.sum(latest_deaths[key])):.0f}"
                    )
                print(f"[MM-PLAY] steps={steps} | " + " | ".join(parts), flush=True)
    except KeyboardInterrupt:
        print("\n[MM-PLAY] Interrompido pelo usuario. Gerando placar parcial.", flush=True)

    results: Dict[str, Dict[str, float]] = {}
    print("[MM-PLAY] Placar final por agente:", flush=True)
    for rt in groups:
        key = rt.spec.cfg_path
        frags = float(np.sum(latest_frags[key]))
        deaths = float(np.sum(latest_deaths[key]))
        actors = float(rt.env.num_envs)
        results[key] = {
            "actors": actors,
            "steps": float(steps),
            "frags": frags,
            "deaths": deaths,
            "reward_total": float(reward_totals[key]),
        }
        print(
            f"[MM-PLAY][RESULT] {key} model={rt.agent_cfg.model_name} "
            f"actors={int(actors)} steps={steps} frags={frags:.0f} "
            f"deaths={deaths:.0f} frags/actor={frags / max(1.0, actors):.3f} "
            f"reward_total={reward_totals[key]:.3f}",
            flush=True,
        )

    if len(groups) >= 2:
        winner = max(groups, key=lambda rt: results[rt.spec.cfg_path]["frags"])
        print(f"[MM-PLAY] Melhor por frags: {winner.spec.cfg_path}", flush=True)

    _append_play_results_csv(groups, results)


# ======================================================================
# main()
# ======================================================================

def main() -> None:
    args = parse_args()

    agent_specs: List[AgentGroupSpec] = [parse_agent_spec(s) for s in args.agent]

    for spec in agent_specs:
        print(f"[MM-TRAIN] Grupo: {spec.cfg_path} x {spec.count}")

    total_actors_per_match = sum(spec.count for spec in agent_specs)
    match_plan = expand_match_plan(args)
    args.num_matches = len(match_plan)
    total_actors = total_actors_per_match * len(match_plan)
    print(
        f"[MM-TRAIN] Topologia: {len(match_plan)} partidas, "
        f"{total_actors_per_match} atores por partida, "
        f"{total_actors} atores no total."
    )

    if getattr(args, "match", None):
        print("[MM-TRAIN] Match plan:")
        for idx, match in enumerate(match_plan):
            print(
                f"  [{idx}] scenario={match.scenario!r} "
                f"map={match.map_name!r} wad={match.wad!r}"
            )
    else:
        print(f"[MM-TRAIN] Map={args.map!r} | Wad={args.wad!r}")

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
            shm_obs=bool(args.shm_obs),
            require_existing_models=bool(args.play),
            warmstart_reset_steps=bool(args.warmstart_reset_steps),
        )

        # 4) Treino ou play multi-modelo
        if bool(args.play):
            play_multi_models(groups, play_steps=int(args.play_steps))
        else:
            train_multi_models(
                groups,
                chunk_steps=args.chunk_steps,
                progress_bar=bool(args.progress_bar),
            )

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

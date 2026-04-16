#!/usr/bin/env python3
"""
Launcher unificado para treino distribuído de VizDoom.

Modos:

1) SINGLE-MODEL (framework.distributed_train):
   - Um único YAML, com vários atores compartilhando o mesmo modelo.

2) MULTI-MODEL (framework.distributed_train_multi):
   - Vários YAMLs diferentes, cada um com sua própria contagem.

Extras:
- --map: nome do mapa (ex: map01, MAP01)
- --wad: arquivo .wad/.pk3 (nome dentro de framework/maps/ ou caminho completo)

Novo:
- --shm-obs: usa shared memory para observações (mesma máquina). Requer suporte no trainer/actor.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import Dict, Tuple, List, Optional


# ----------------------------------------------------------------------
# Parsing de specs YAML:COUNT
# ----------------------------------------------------------------------
def parse_agent_spec(spec: str) -> Tuple[str, int]:
    """
    Formato aceito: "caminho.yaml:count"
    Ex: "example_agent.yaml:3"

    Observação (Windows): caminhos absolutos têm ":" (ex: C:\\x\\a.yaml:3),
    então usamos split pela ÚLTIMA ocorrência.
    """
    if ":" not in spec:
        raise ValueError(
            f"Formato inválido para --agent '{spec}'. "
            f"Use: caminho.yaml:count (ex.: example_agent.yaml:3)"
        )

    cfg_path, count_str = spec.rsplit(":", 1)
    cfg_path = cfg_path.strip()
    count_str = count_str.strip()

    if not cfg_path:
        raise ValueError(f"YAML vazio em --agent '{spec}'.")

    try:
        count = int(count_str)
    except ValueError as e:
        raise ValueError(f"COUNT inválido em --agent '{spec}' (esperado int).") from e

    if count <= 0:
        raise ValueError(f"COUNT deve ser > 0 em --agent '{spec}'.")

    return cfg_path, count


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launcher unificado para treino distribuído VizDoom (single ou multi-modelo)."
    )

    # Modo single: um YAML explícito
    parser.add_argument(
        "--cfg",
        help="YAML do agente (modo single-model). Se não for fornecido, use --agent YAML:COUNT.",
    )

    # Modo multi / alternativa para single: vários YAML:COUNT
    parser.add_argument(
        "--agent",
        action="append",
        help="Especificação de agente: caminho_yaml:count (pode repetir). "
             "Ex: --agent example_agent.yaml:4 --agent bob.yaml:4",
    )

    parser.add_argument("--num-matches", type=int, default=1)
    parser.add_argument(
        "--actors-per-match",
        type=int,
        help="Número de jogadores por partida (somente modo single-model). "
             "Obrigatório se você usar --cfg sem --agent.",
    )

    parser.add_argument("--port", type=int, default=5029)
    parser.add_argument("--game-ip", default="127.0.0.1")
    parser.add_argument("--timelimit", type=float, default=0.0)
    parser.add_argument("--stack", type=int, default=4)

    parser.add_argument(
        "--render",
        choices=["none", "host", "host_agent", "all"],
        default="host",
        help="Renderização: none (sem janela), host (só host), all (todos).",
    )

    parser.add_argument("--trainer-port", type=int, default=7000)
    parser.add_argument("--auth-key", default="vizdoom_dm")
    parser.add_argument("--chunk-steps", type=int, default=50_000)
    parser.add_argument("--host-start-delay", type=float, default=1.5)
    parser.add_argument("--actor-start-delay", type=float, default=0.05)
    parser.add_argument(
        "--scenario",
        default=None,
        help="Cenário base .cfg/.wad/.pk3 em framework/maps ou caminho completo.",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=8,
        help="Frames repetidos por ação. Menor = visual mais suave, maior = treino mais rápido.",
    )
    parser.add_argument(
        "--ticrate",
        type=int,
        default=30,
        help="Tickrate do motor. Maior = simulação/render mais suaves, mas mais pesados.",
    )

    # Mapa e wad/pk3
    parser.add_argument("--map", default="map01", help="Nome do mapa (ex: map01, MAP01).")
    parser.add_argument(
        "--wad",
        default=None,
        help="WAD/PK3: nome dentro de framework/maps (ex: mypack.wad) ou caminho completo.",
    )

    # Shared memory para observações
    parser.add_argument(
        "--shm-obs",
        action="store_true",
        help="Usa shared memory para observações (mesma máquina). Requer suporte no trainer/actor.",
    )

    parser.add_argument(
        "--warmstart-reset-steps",
        action="store_true",
        help="Carrega so os pesos do modelo salvo e reinicia steps/schedules do zero.",
    )

    return parser.parse_args()


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _maybe_add_scenario_map_and_wad(
    cmd: List[str],
    scenario: Optional[str],
    map_name: str,
    wad: Optional[str],
) -> None:
    if scenario is not None:
        scenario = str(scenario).strip()
        if scenario:
            cmd += ["--scenario", scenario]
    cmd += ["--map", str(map_name)]
    if wad is not None:
        wad = str(wad).strip()
        if wad:
            cmd += ["--wad", wad]


def _apply_render_flags(cmd: List[str], render: str) -> None:
    if render == "all":
        cmd.append("--render-all")
    elif render == "host_agent":
        cmd.append("--render-host-agent")
    elif render == "host":
        cmd.append("--render-host")
    # render == "none" -> sem flag


def _maybe_add_shm_obs(cmd: List[str], shm_obs: bool) -> None:
    if shm_obs:
        cmd.append("--shm-obs")


# ----------------------------------------------------------------------
# Construção dos comandos
# ----------------------------------------------------------------------
def build_single_model_cmd_from_cfg(args: argparse.Namespace) -> List[str]:
    if args.actors_per_match is None or args.actors_per_match <= 0:
        raise ValueError(
            "Para modo single-model com --cfg, você deve informar --actors-per-match > 0."
        )

    total_actors = args.num_matches * args.actors_per_match

    cmd: List[str] = [
        sys.executable,
        "-m",
        "framework.distributed_train",
        "--cfg",
        args.cfg,
        "--num-matches",
        str(args.num_matches),
        "--actors-per-match",
        str(args.actors_per_match),
        "--num-actors",
        str(total_actors),
        "--game-port",
        str(args.port),
        "--game-ip",
        args.game_ip,
        "--timelimit",
        str(args.timelimit),
        "--stack",
        str(args.stack),
        "--trainer-host",
        "127.0.0.1",
        "--trainer-port",
        str(args.trainer_port),
        "--auth-key",
        args.auth_key,
        "--host-start-delay",
        str(args.host_start_delay),
        "--actor-start-delay",
        str(args.actor_start_delay),
        "--frame-skip",
        str(args.frame_skip),
        "--ticrate",
        str(args.ticrate),
    ]

    _maybe_add_scenario_map_and_wad(cmd, args.scenario, args.map, args.wad)
    _apply_render_flags(cmd, args.render)
    _maybe_add_shm_obs(cmd, bool(args.shm_obs))
    if args.warmstart_reset_steps:
        cmd.append("--warmstart-reset-steps")
    return cmd


def build_single_model_cmd_from_agents(args: argparse.Namespace) -> List[str]:
    if not args.agent:
        raise ValueError("build_single_model_cmd_from_agents: args.agent vazio.")

    yaml_to_count: Dict[str, int] = {}
    for spec in args.agent:
        cfg_path, count = parse_agent_spec(spec)
        yaml_to_count[cfg_path] = yaml_to_count.get(cfg_path, 0) + count

    if len(yaml_to_count) != 1:
        raise RuntimeError(
            "build_single_model_cmd_from_agents chamado mas há mais de um YAML em --agent."
        )

    cfg_path = next(iter(yaml_to_count.keys()))
    actors_per_match = next(iter(yaml_to_count.values()))

    if args.actors_per_match is not None and args.actors_per_match != actors_per_match:
        raise ValueError(
            f"--actors-per-match={args.actors_per_match} conflita com a soma de counts em --agent ({actors_per_match})."
        )

    args_local = argparse.Namespace(**vars(args))
    args_local.cfg = cfg_path
    args_local.actors_per_match = actors_per_match
    return build_single_model_cmd_from_cfg(args_local)


def build_multi_model_cmd(args: argparse.Namespace) -> List[str]:
    if not args.agent:
        raise ValueError("Modo multi-model requer pelo menos um --agent YAML:COUNT.")

    cmd: List[str] = [
        sys.executable,
        "-m",
        "framework.distributed_train_multi",
        "--num-matches",
        str(args.num_matches),
        "--game-port",
        str(args.port),
        "--game-ip",
        args.game_ip,
        "--timelimit",
        str(args.timelimit),
        "--stack",
        str(args.stack),
        "--trainer-host",
        "127.0.0.1",
        "--trainer-port",
        str(args.trainer_port),
        "--auth-key",
        args.auth_key,
        "--chunk-steps",
        str(args.chunk_steps),
        "--host-start-delay",
        str(args.host_start_delay),
        "--actor-start-delay",
        str(args.actor_start_delay),
        "--frame-skip",
        str(args.frame_skip),
        "--ticrate",
        str(args.ticrate),
    ]

    for spec in args.agent:
        cmd.extend(["--agent", spec])

    _maybe_add_scenario_map_and_wad(cmd, args.scenario, args.map, args.wad)
    _apply_render_flags(cmd, args.render)
    _maybe_add_shm_obs(cmd, bool(args.shm_obs))
    if args.warmstart_reset_steps:
        cmd.append("--warmstart-reset-steps")
    return cmd


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    if args.cfg and args.agent:
        raise ValueError(
            "Use EITHER --cfg (single-model) OR --agent (single ou multi-model). Não use os dois."
        )

    if not args.cfg and not args.agent:
        raise ValueError(
            "Você deve informar --cfg (single-model) ou pelo menos um --agent YAML:COUNT."
        )

    if args.cfg:
        train_cmd = build_single_model_cmd_from_cfg(args)
    else:
        yaml_to_count: Dict[str, int] = {}
        for spec in args.agent:
            cfg_path, count = parse_agent_spec(spec)
            yaml_to_count[cfg_path] = yaml_to_count.get(cfg_path, 0) + count

        if len(yaml_to_count) == 1:
            train_cmd = build_single_model_cmd_from_agents(args)
        else:
            train_cmd = build_multi_model_cmd(args)

    print("[RUN_TRAIN] Comando final:")
    print(" ", " ".join(train_cmd))

    env = os.environ.copy()
    if args.shm_obs:
        env["VIZDM_SHM_OBS"] = "1"

    try:
        subprocess.run(train_cmd, check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"[RUN_TRAIN] Erro ao executar treino: {e}")
        raise
    except KeyboardInterrupt:
        print("\n[RUN_TRAIN] Interrompido pelo usuário.")


if __name__ == "__main__":
    main()

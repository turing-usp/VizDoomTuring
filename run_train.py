#!/usr/bin/env python3
"""
Launcher unificado para treino distribuido de VizDoom.

Modos:

1) SINGLE-MODEL (framework.distributed_train):
   - Um unico YAML, com varios atores compartilhando o mesmo modelo.

2) MULTI-MODEL (framework.distributed_train_multi):
   - Varios YAMLs diferentes, cada um com sua propria contagem.

Extras:
- --map: nome do mapa (ex: map01, MAP01)
- --wad: arquivo .wad/.pk3 (nome dentro de framework/maps/ ou caminho completo)
- --match: mistura de partidas por mapa/cenario, no formato SCENARIO|MAP|COUNT
- --train-cfg: YAML separado para configurar uma sessao de treino
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

import yaml


def parse_agent_spec(spec: str) -> Tuple[str, int]:
    """
    Formato aceito: "caminho.yaml:count"
    Ex: "example_agent.yaml:3"

    Observacao (Windows): caminhos absolutos tem ":" (ex: C:\\x\\a.yaml:3),
    entao usamos split pela ULTIMA ocorrencia.
    """
    if ":" not in spec:
        raise ValueError(
            f"Formato invalido para --agent '{spec}'. "
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
        raise ValueError(f"COUNT invalido em --agent '{spec}' (esperado int).") from e

    if count <= 0:
        raise ValueError(f"COUNT deve ser > 0 em --agent '{spec}'.")

    return cfg_path, count


def parse_match_spec(spec: str) -> Tuple[Optional[str], str, int, Optional[str]]:
    """
    Formatos aceitos:
      SCENARIO|MAP|COUNT
      SCENARIO|MAP|COUNT|WAD

    Exemplos:
      multi_duel_dm15_smart.wad|map01|6
      multi_duel_dm15.wad|map01|4
      custom.cfg|map02|2|bonus_pack.wad
    """
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

    return scenario, map_name, count, wad


def _sum_match_counts(match_specs: Optional[List[str]]) -> int:
    if not match_specs:
        return 0
    total = 0
    for spec in match_specs:
        _scenario, _map_name, count, _wad = parse_match_spec(spec)
        total += count
    return total


def _load_yaml_dict(path: str) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Arquivo YAML invalido: {path}")
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launcher unificado para treino distribuido VizDoom (single ou multi-modelo)."
    )

    parser.add_argument(
        "--cfg",
        help="YAML do agente (modo single-model). Se nao for fornecido, use --agent YAML:COUNT.",
    )
    parser.add_argument(
        "--train-cfg",
        help="YAML de configuracao do treino (separado do YAML do modelo).",
    )

    parser.add_argument(
        "--agent",
        action="append",
        help="Especificacao de agente: caminho_yaml:count (pode repetir). "
        "Ex: --agent example_agent.yaml:4 --agent bob.yaml:4",
    )

    parser.add_argument("--num-matches", type=int, default=1)
    parser.add_argument(
        "--actors-per-match",
        type=int,
        help="Numero de jogadores por partida (somente modo single-model). "
        "Obrigatorio se voce usar --cfg sem --agent.",
    )

    parser.add_argument("--port", type=int, default=5029)
    parser.add_argument("--game-ip", default="127.0.0.1")
    parser.add_argument("--timelimit", type=float, default=0.0)
    parser.add_argument("--stack", type=int, default=4)

    parser.add_argument(
        "--render",
        choices=["none", "host", "host_agent", "all"],
        default="host",
        help="Renderizacao: none (sem janela), host (so host), all (todos).",
    )

    parser.add_argument("--trainer-port", type=int, default=7000)
    parser.add_argument("--auth-key", default="vizdoom_dm")
    parser.add_argument("--chunk-steps", type=int, default=50_000)
    parser.add_argument("--host-start-delay", type=float, default=1.5)
    parser.add_argument("--actor-start-delay", type=float, default=0.05)
    parser.add_argument(
        "--scenario",
        default=None,
        help="Cenario base .cfg/.wad/.pk3 em framework/maps ou caminho completo.",
    )
    parser.add_argument(
        "--match",
        action="append",
        help="Mistura de partidas: SCENARIO|MAP|COUNT ou SCENARIO|MAP|COUNT|WAD. Pode repetir.",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=8,
        help="Frames repetidos por acao. Menor = visual mais suave, maior = treino mais rapido.",
    )
    parser.add_argument(
        "--ticrate",
        type=int,
        default=30,
        help="Tickrate do motor. Maior = simulacao/render mais suaves, mas mais pesados.",
    )

    parser.add_argument("--map", default="map01", help="Nome do mapa (ex: map01, MAP01).")
    parser.add_argument(
        "--wad",
        default=None,
        help="WAD/PK3: nome dentro de framework/maps (ex: mypack.wad) ou caminho completo.",
    )

    parser.add_argument(
        "--shm-obs",
        action="store_true",
        help="Usa shared memory para observacoes (mesma maquina). Requer suporte no trainer/actor.",
    )
    parser.add_argument(
        "--warmstart-reset-steps",
        action="store_true",
        help="Carrega so os pesos do modelo salvo e reinicia steps/schedules do zero.",
    )
    parser.add_argument(
        "--progress-bar",
        action="store_true",
        help="Mostra a barra de progresso do SB3 durante o treino.",
    )
    parser.add_argument(
        "--play",
        action="store_true",
        help="Carrega o modelo e apenas assiste/joga, sem treinar nem salvar.",
    )
    parser.add_argument(
        "--play-steps",
        type=int,
        default=0,
        help="Quantidade de decisoes no modo --play (0 = ate Ctrl+C).",
    )

    return parser.parse_args()


def _apply_train_cfg(args: argparse.Namespace) -> argparse.Namespace:
    if not args.train_cfg:
        return args

    train_cfg = _load_yaml_dict(args.train_cfg)
    defaults = {
        "cfg": None,
        "num_matches": 1,
        "actors_per_match": None,
        "port": 5029,
        "game_ip": "127.0.0.1",
        "timelimit": 0.0,
        "stack": 4,
        "render": "host",
        "trainer_port": 7000,
        "auth_key": "vizdoom_dm",
        "chunk_steps": 50_000,
        "host_start_delay": 1.5,
        "actor_start_delay": 0.05,
        "scenario": None,
        "frame_skip": 8,
        "ticrate": 30,
        "map": "map01",
        "wad": None,
        "shm_obs": False,
        "warmstart_reset_steps": False,
        "progress_bar": False,
        "play": False,
        "play_steps": 0,
    }

    for key in defaults:
        if key in train_cfg and getattr(args, key) == defaults[key]:
            setattr(args, key, train_cfg[key])

    if args.match is None and "matches" in train_cfg:
        matches = train_cfg.get("matches") or []
        if not isinstance(matches, list):
            raise ValueError("'matches' no train-cfg deve ser uma lista.")

        fallback_scenario = train_cfg.get("scenario")
        fallback_map = train_cfg.get("map", defaults["map"])
        fallback_wad = train_cfg.get("wad")

        built_specs: List[str] = []
        for idx, item in enumerate(matches):
            if isinstance(item, str):
                parse_match_spec(item)
                built_specs.append(item)
                continue

            if not isinstance(item, dict):
                raise ValueError(f"matches[{idx}] invalido no train-cfg.")

            scenario = item.get("scenario", fallback_scenario)
            map_name = item.get("map", fallback_map)
            count = item.get("count")
            wad = item.get("wad", fallback_wad)

            if count is None:
                raise ValueError(f"matches[{idx}] precisa informar 'count'.")

            spec = f"{'' if scenario is None else scenario}|{map_name}|{count}"
            if wad is not None:
                spec += f"|{wad}"
            parse_match_spec(spec)
            built_specs.append(spec)

        args.match = built_specs

    if args.match:
        args.num_matches = _sum_match_counts(args.match)

    return args


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


def _maybe_add_match_specs(cmd: List[str], match_specs: Optional[List[str]]) -> None:
    if not match_specs:
        return
    for spec in match_specs:
        cmd += ["--match", str(spec)]


def _apply_render_flags(cmd: List[str], render: str) -> None:
    if render == "all":
        cmd.append("--render-all")
    elif render == "host_agent":
        cmd.append("--render-host-agent")
    elif render == "host":
        cmd.append("--render-host")


def _maybe_add_shm_obs(cmd: List[str], shm_obs: bool) -> None:
    if shm_obs:
        cmd.append("--shm-obs")


def build_single_model_cmd_from_cfg(args: argparse.Namespace) -> List[str]:
    if args.actors_per_match is None or args.actors_per_match <= 0:
        raise ValueError(
            "Para modo single-model com --cfg, voce deve informar --actors-per-match > 0."
        )

    if args.match:
        args.num_matches = _sum_match_counts(args.match)

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

    if args.match:
        _maybe_add_match_specs(cmd, args.match)
    else:
        _maybe_add_scenario_map_and_wad(cmd, args.scenario, args.map, args.wad)
    _apply_render_flags(cmd, args.render)
    _maybe_add_shm_obs(cmd, bool(args.shm_obs))
    if args.warmstart_reset_steps:
        cmd.append("--warmstart-reset-steps")
    if args.progress_bar:
        cmd.append("--progress-bar")
    if args.play:
        cmd.append("--play")
    if int(args.play_steps) > 0:
        cmd += ["--play-steps", str(args.play_steps)]
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
            "build_single_model_cmd_from_agents chamado mas ha mais de um YAML em --agent."
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
    if args.match:
        raise ValueError("Mistura de mapas (--match / train-cfg matches) ainda nao e suportada no modo multi-model.")

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
    if args.progress_bar:
        cmd.append("--progress-bar")
    if args.play:
        cmd.append("--play")
    if int(args.play_steps) > 0:
        cmd += ["--play-steps", str(args.play_steps)]
    return cmd


def main() -> None:
    args = _apply_train_cfg(parse_args())

    if args.cfg and args.agent:
        raise ValueError(
            "Use EITHER --cfg (single-model) OR --agent (single ou multi-model). Nao use os dois."
        )

    if not args.cfg and not args.agent:
        raise ValueError(
            "Voce deve informar --cfg (single-model) ou pelo menos um --agent YAML:COUNT."
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
        print("\n[RUN_TRAIN] Interrompido pelo usuario.")


if __name__ == "__main__":
    main()

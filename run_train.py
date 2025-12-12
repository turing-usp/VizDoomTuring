#!/usr/bin/env python3
"""
Launcher unificado para treino distribuído de VizDoom.

Modos:

1) SINGLE-MODEL (normal, usa framework.distributed_train):
   - Um único YAML, com vários atores compartilhando o mesmo modelo.
   - Formas de uso:

       a) Usando --cfg:
           python run_train.py \
             --cfg example_agent.yaml \
             --num-matches 1 \
             --actors-per-match 8 \
             --render host

       b) Usando --agent com apenas 1 YAML:
           python run_train.py \
             --agent example_agent.yaml:8 \
             --num-matches 1 \
             --render host

   Em ambos os casos, é chamado:
       python -m framework.distributed_train ...

2) MULTI-MODEL (usa framework.distributed_train_multi):
   - Vários YAMLs diferentes, cada um com sua própria contagem.
   - Exemplo:

       python run_train.py \
         --agent alice.yaml:4 \
         --agent bob.yaml:4 \
         --num-matches 1 \
         --render host

   Isso chama:
       python -m framework.distributed_train_multi ...
"""

import argparse
import subprocess
import sys
from typing import Dict, Tuple, List


# ----------------------------------------------------------------------
# Parsing de specs YAML:COUNT
# ----------------------------------------------------------------------


def parse_agent_spec(spec: str) -> Tuple[str, int]:
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
        help="YAML do agente (modo single-model). "
             "Se não for fornecido, você pode usar --agent YAML:COUNT.",
    )

    # Modo multi / alternativa para single: vários YAML:COUNT
    parser.add_argument(
        "--agent",
        action="append",
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
        "--actors-per-match",
        type=int,
        help=(
            "Número de jogadores por partida (somente modo single-model). "
            "Obrigatório se você usar --cfg sem --agent."
        ),
    )

    parser.add_argument(
        "--port",
        type=int,
        default=5029,
        help="Porta base do servidor VizDoom (default: 5029).",
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
        help="Frames empilhados na entrada da rede (default: 4).",
    )

    parser.add_argument(
        "--render",
        choices=["none", "host", "all"],
        default="host",
        help="Renderização: none (sem janela), host (só host), all (todos).",
    )

    parser.add_argument(
        "--trainer-port",
        type=int,
        default=7000,
        help="Porta IPC treinador<->atores (default: 7000).",
    )

    parser.add_argument(
        "--auth-key",
        default="vizdoom_dm",
        help="Chave IPC treinador<->atores (default: vizdoom_dm).",
    )

    parser.add_argument(
        "--chunk-steps",
        type=int,
        default=50_000,
        help="Steps por chunk de treino (apenas multi-modelo).",
    )

    return parser.parse_args()


# ----------------------------------------------------------------------
# Construção dos comandos
# ----------------------------------------------------------------------


def build_single_model_cmd_from_cfg(args: argparse.Namespace) -> List[str]:
    """
    Usa --cfg e --actors-per-match para montar comando do trainer normal.
    """
    if args.actors_per_match is None or args.actors_per_match <= 0:
        raise ValueError(
            "Para modo single-model com --cfg, você deve informar "
            "--actors-per-match > 0."
        )

    total_actors = args.num_matches * args.actors_per_match

    print(
        f"[RUN_TRAIN] MODO SINGLE-MODEL (distributed_train): "
        f"{args.num_matches} partidas x {args.actors_per_match} jogadores "
        f"= {total_actors} atores."
    )

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
        # distributed_train calcula num_actors internamente,
        # mas podemos ainda passar o valor calculado, se quiser:
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
    ]

    if args.render == "all":
        cmd.append("--render-all")
    elif args.render == "host":
        cmd.append("--render-host")
    # render == "none" => sem flag

    return cmd


def build_single_model_cmd_from_agents(args: argparse.Namespace) -> List[str]:
    """
    Usa --agent YAML:COUNT, mas com apenas 1 YAML único.
    Converte isso em um modo single-model (distributed_train).
    """
    yaml_to_count: Dict[str, int] = {}

    for spec in args.agent:
        cfg_path, count = parse_agent_spec(spec)
        yaml_to_count[cfg_path] = yaml_to_count.get(cfg_path, 0) + count

    if len(yaml_to_count) != 1:
        raise RuntimeError(
            "build_single_model_cmd_from_agents chamado mas há "
            "mais de um YAML em --agent."
        )

    cfg_path = next(iter(yaml_to_count.keys()))
    actors_per_match = next(iter(yaml_to_count.values()))

    # Se o usuário tiver passado --actors-per-match também, podemos checar consistência:
    if args.actors_per_match is not None and args.actors_per_match != actors_per_match:
        raise ValueError(
            f"--actors-per-match={args.actors_per_match} conflita com a soma de "
            f"counts em --agent ({actors_per_match}). Use só um ou alinhe os valores."
        )

    args_local = argparse.Namespace(**vars(args))
    args_local.cfg = cfg_path
    args_local.actors_per_match = actors_per_match

    return build_single_model_cmd_from_cfg(args_local)


def build_multi_model_cmd(args: argparse.Namespace) -> List[str]:
    """
    Usa vários --agent YAML:COUNT para chamar framework.distributed_train_multi.
    """
    print(
        "[RUN_TRAIN] MODO MULTI-MODEL (distributed_train_multi); "
        "YAMLs e counts recebidos:"
    )
    for spec in args.agent:
        print(f"    --agent {spec}")

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
    ]

    for spec in args.agent:
        cmd.extend(["--agent", spec])

    if args.render == "all":
        cmd.append("--render-all")
    elif args.render == "host":
        cmd.append("--render-host")
    # render == "none" => sem flag

    return cmd


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    # Validação básica
    if args.cfg and args.agent:
        raise ValueError(
            "Use EITHER --cfg (single-model) OR --agent (single ou multi-model). "
            "Não use os dois ao mesmo tempo."
        )

    if not args.cfg and not args.agent:
        raise ValueError(
            "Você deve informar --cfg (single-model) ou pelo menos um --agent YAML:COUNT."
        )

    # Decisão de modo
    if args.cfg:
        # Modo single-model clássico: um YAML, vários atores
        train_cmd = build_single_model_cmd_from_cfg(args)

    else:
        # Só temos --agent
        yaml_to_count: Dict[str, int] = {}
        for spec in args.agent:
            cfg_path, count = parse_agent_spec(spec)
            yaml_to_count[cfg_path] = yaml_to_count.get(cfg_path, 0) + count

        if len(yaml_to_count) == 1:
            # Mesmo com --agent, só um YAML => single-model
            train_cmd = build_single_model_cmd_from_agents(args)
        else:
            # Vários YAMLs diferentes => multi-model
            train_cmd = build_multi_model_cmd(args)

    print("[RUN_TRAIN] Comando final:")
    print(" ", " ".join(train_cmd))

    try:
        subprocess.run(train_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[RUN_TRAIN] Erro ao executar treino: {e}")
    except KeyboardInterrupt:
        print("\n[RUN_TRAIN] Interrompido pelo usuário.")


if __name__ == "__main__":
    main()

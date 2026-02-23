#!/usr/bin/env python3
"""
Launcher para jogar vários agentes diferentes (multi-modelo) na mesma partida.

Uso exemplo (1 partida, 4 jogadores: 2 Alice, 2 Bob):

    python -m vizdm_comp.run_multi_play \
        --agent example_agent.yaml:2 \
        --agent bob.yaml:2 \
        --num-matches 1 \
        --port 5029 \
        --render host

Regras:
- Cada `--agent` é da forma: YAML_PATH:COUNT
- O primeiro agente listado fornece o HOST da partida.
- Todos os YAMLs devem ter `train: false` para modo "play" (sem treino).
- O ticrate "real" (35) é aplicado porque `train: false` (vide env.py).
"""

import argparse
import subprocess
import sys
from dataclasses import dataclass
from typing import List


@dataclass
class AgentSpec:
    cfg_path: str
    count: int


def parse_agent_spec(spec: str) -> AgentSpec:
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
    return AgentSpec(cfg_path=cfg_path, count=count)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launcher para jogar múltiplos agentes (multi-modelo) no VizDoom DM."
    )

    parser.add_argument(
        "--agent",
        action="append",
        required=True,
        help="Especificação de agente: caminho_yaml:count (pode repetir). "
             "Ex: --agent example_agent.yaml:2 --agent bob.yaml:2",
    )

    parser.add_argument(
        "--num-matches",
        type=int,
        default=1,
        help="Número de partidas em paralelo (default: 1).",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=5029,
        help="Porta base do servidor VizDoom (default: 5029). "
             "Cada partida usa port+idx.",
    )

    parser.add_argument(
        "--ip",
        default="127.0.0.1",
        help="IP do host VizDoom (default: 127.0.0.1).",
    )

    parser.add_argument(
        "--timelimit",
        type=float,
        default=10.0,
        help="Duração da partida em minutos (0 = infinito). (default: 10.0)",
    )

    parser.add_argument(
        "--stack",
        type=int,
        default=4,
        help="Frames empilhados no cliente (default: 4).",
    )

    parser.add_argument(
        "--render",
        choices=["none", "host", "all"],
        default="host",
        help="Renderização: none (sem janela), host (só host), all (todos).",
    )

    return parser.parse_args()


def build_client_cmd(
    cfg_path: str,
    total_players: int,
    port: int,
    ip: str,
    timelimit: float,
    stack: int,
    is_host: bool,
    render: bool,
) -> List[str]:
    """
    Monta comando para lançar um processo de cliente (framework.client).
    Supõe que client.py aceite:
        --cfg, --port, --ip, --players, --timelimit, --stack, --host, --render
    """
    cmd: List[str] = [
        sys.executable,
        "-m",
        "vizdm_comp.framework.client",
        "--cfg",
        cfg_path,
        "--port",
        str(port),
        "--ip",
        ip,
        "--players",
        str(total_players),
        "--timelimit",
        str(timelimit),
        "--stack",
        str(stack),
    ]

    if is_host:
        cmd.append("--host")
        if render:
            cmd.append("--render")
    else:
        if render:
            cmd.append("--render")

    return cmd


def main() -> None:
    args = parse_args()

    agent_specs: List[AgentSpec] = [parse_agent_spec(s) for s in args.agent]
    if len(agent_specs) == 0:
        raise RuntimeError("É necessário pelo menos um --agent YAML:count.")

    # Primeiro agente da lista será o HOST da partida
    host_spec = agent_specs[0]

    total_players_per_match = sum(a.count for a in agent_specs)
    print(
        f"[PLAY] Topologia: {args.num_matches} partida(s), "
        f"{total_players_per_match} jogadores por partida."
    )
    print("[PLAY] Agentes por partida:")
    for spec in agent_specs:
        print(f"   - {spec.cfg_path}: {spec.count} jogadores")

    all_procs: List[subprocess.Popen] = []

    try:
        for match_idx in range(args.num_matches):
            match_port = args.port + match_idx
            print(f"\n[PLAY] === Partida {match_idx} (porta {match_port}) ===")

            # 1) HOST: usa o primeiro agente
            print(f"[PLAY] Lançando HOST com {host_spec.cfg_path} ...")
            host_cmd = build_client_cmd(
                cfg_path=host_spec.cfg_path,
                total_players=total_players_per_match,
                port=match_port,
                ip=args.ip,
                timelimit=args.timelimit,
                stack=args.stack,
                is_host=True,
                render=(args.render in ("host", "all")),
            )
            print("[PLAY][CMD]", " ".join(host_cmd))
            all_procs.append(subprocess.Popen(host_cmd))

            # Pequeno delay para o host subir
            # (se quiser, ajuste esse valor depois de testar)
            import time
            time.sleep(5.0)

            # 2) CLIENTES: percorre todos os AgentSpec, respeitando counts,
            #    mas já usamos 1 jogador do host_spec.
            print("[PLAY] Lançando CLIENTES ...")
            for spec_idx, spec in enumerate(agent_specs):
                players_for_this_spec = spec.count

                # Se for o mesmo spec do host, já usamos 1 instância
                start = 0
                if spec_idx == 0:
                    players_for_this_spec -= 1
                    start = 1  # só para log

                for local_idx in range(start, start + players_for_this_spec):
                    print(
                        f"[PLAY]   Cliente ({spec.cfg_path}) "
                        f"local_idx={local_idx}"
                    )
                    client_cmd = build_client_cmd(
                        cfg_path=spec.cfg_path,
                        total_players=total_players_per_match,
                        port=match_port,
                        ip=args.ip,
                        timelimit=args.timelimit,
                        stack=args.stack,
                        is_host=False,
                        render=(args.render == "all"),
                    )
                    print("[PLAY][CMD]", " ".join(client_cmd))
                    all_procs.append(subprocess.Popen(client_cmd))

        print("\n[PLAY] Todos os processos lançados. Pressione Ctrl+C para encerrar.")
        # Espera infinito (até Ctrl+C)
        for p in all_procs:
            p.wait()

    except KeyboardInterrupt:
        print("\n[PLAY] Interrompido pelo usuário (Ctrl+C).")
    finally:
        print("[PLAY] Encerrando processos...")
        for p in all_procs:
            try:
                p.terminate()
            except Exception:
                pass


if __name__ == "__main__":
    main()

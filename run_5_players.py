#!/usr/bin/env python3
import subprocess
import sys


def build_base_cmd() -> list[str]:
    """
    Comando base para chamar o framework.client com o seu example_agent.yaml.
    Ajuste o caminho do YAML se ele não estiver na raiz.
    """
    return [
        sys.executable,
        "-m",
        "framework.client",
        "--cfg",
        "example_agent.yaml",
        "--players",
        "5",
        "--port",
        "5029",
        "--timelimit",
        "3",
        "--stack",
        "4",
    ]


def main() -> None:
    procs: list[subprocess.Popen] = []

    base_cmd = build_base_cmd()

    # 1) Host (jogador 1) - sobe a partida
    host_cmd = base_cmd + ["--host", "--render"]
    print("Iniciando HOST:", " ".join(host_cmd))
    procs.append(subprocess.Popen(host_cmd))

    # 2) Clientes (jogadores 2..5) - se conectam ao host
    for i in range(4):
        client_cmd = base_cmd + ["--ip", "127.0.0.1", "--render"]
        print(f"Iniciando CLIENTE {i+2}:", " ".join(client_cmd))
        procs.append(subprocess.Popen(client_cmd))

    # Espera todos terminarem (Ctrl+C encerra)
    try:
        for p in procs:
            p.wait()
    except KeyboardInterrupt:
        print("Interrompido, encerrando processos...")
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import yaml
import os
import traceback
import multiprocessing as mp

# Importações locais (mantendo compatibilidade com seu config.py)
from .config import (
    DMConfig, 
    AgentConfig, 
    EngineRewardConfig, 
    ShapingConfig, 
    PolicyConfig
)

# ==============================================================================
# 1. Função usada pelo TREINADOR (distributed_train_multi.py)
# ==============================================================================
def load_agent_cfg(yaml_path: str) -> AgentConfig:
    """
    Lê o YAML e retorna um objeto AgentConfig.
    O distributed_train_multi.py precisa disso para saber criar a rede neural.
    """
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"YAML não encontrado: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)
    
    # Tratamento seguro de dicionários vazios
    er = EngineRewardConfig(**y.get("engine_reward", {}))
    sh = ShapingConfig(**y.get("shaping", {}))
    pol = PolicyConfig(**y.get("policy", {}))
    
    # Tratamento de render_settings (caso exista no YAML)
    from .config import RenderSettingsConfig
    render_settings = RenderSettingsConfig(**y.get("render_settings", {}))

    return AgentConfig(
        name=y.get("name", "Client"),
        colorset=y.get("colorset", 3),
        engine_reward=er,
        shaping=sh,
        policy=pol,
        render_settings=render_settings, # Adicionado para garantir compatibilidade
        model_dir=y.get("model_dir", "models"),
        model_name=y.get("model_name", "agent.zip"),
        train=bool(y.get("train", False)),
        train_steps=int(y.get("train_steps", 300_000)),
    )

# ==============================================================================
# 2. Função usada pelo ATOR (distributed_actor.py)
# ==============================================================================
def run_loop_with_env(conn, args, env, agent_cfg):
    """
    Loop principal do Ator Remoto.
    Recebe comandos do Treinador via Socket (conn), executa no Env e devolve a resposta.
    """
    print(f"[CLIENT] Iniciando loop de controle remoto para agente: {agent_cfg.name}")
    
    try:
        while True:
            # 1. Espera comando do Treinador (bloqueante)
            msg = conn.recv()
            
            # Validação básica
            if not isinstance(msg, dict):
                # Ignora pings ou lixo de memória
                continue
            
            cmd = msg.get("cmd")

            # --- COMANDO: RESET ---
            if cmd == "reset":
                # Reseta o ambiente VizDoom
                obs, info = env.reset()
                
                # Garante que o nome do agente vá no info (essencial para o Treinador agrupar)
                if "name" not in info:
                    info["name"] = agent_cfg.name
                
                conn.send({
                    "obs": obs,
                    "info": info
                })

            # --- COMANDO: STEP (Dar um passo) ---
            elif cmd == "step":
                action = msg.get("action")
                
                # Executa a ação no jogo
                # Gymnasium retorna: obs, reward, terminated, truncated, info
                obs, reward, terminated, truncated, info = env.step(action)
                
                # SB3 espera 'done'
                done = terminated or truncated
                
                conn.send({
                    "obs": obs,
                    "reward": reward,
                    "done": done,
                    "info": info
                })

            # --- COMANDO: GET_SPACES (Pegar formato da imagem/ação) ---
            elif cmd == "get_spaces":
                conn.send({
                    "obs_space": env.observation_space,
                    "action_space": env.action_space
                })

            # --- COMANDO: CLOSE ---
            elif cmd == "close":
                print("[CLIENT] Recebi ordem de desconectar.")
                break

            else:
                pass 
                # Comandos desconhecidos são ignorados para não quebrar o loop

    except EOFError:
        print("[CLIENT] Conexão com o Treinador foi encerrada (EOF).")
    except Exception as e:
        print(f"[CLIENT] Erro fatal no loop de execução: {e}")
        traceback.print_exc()
    finally:
        print("[CLIENT] Fechando ambiente e conexão.")
        try:
            env.close()
        except:
            pass
        try:
            conn.close()
        except:
            pass

# ==============================================================================
# 3. Main (Opcional, apenas se quiser rodar localmente sem distribuidor)
# ==============================================================================
if __name__ == "__main__":
    # Esta parte só roda se você chamar 'python -m vizdm_comp.framework.client' direto.
    # No modo distribuído, ela é ignorada.
    
    from .env import DoomDMEnv
    from .train import train_or_play
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, help="YAML do agente")
    ap.add_argument("--port", type=int, default=5029)
    ap.add_argument("--ip", default="127.0.0.1")
    ap.add_argument("--host", action="store_true")
    ap.add_argument("--players", type=int, default=2)
    ap.add_argument("--game_config", default="tag.cfg")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--timelimit", type=float, default=3.0)
    ap.add_argument("--stack", type=int, default=4)
    # Adicionados para compatibilidade com distributed_actor
    ap.add_argument("--wad", default=None)
    ap.add_argument("--map", default="map01")

    args = ap.parse_args()

    agent = load_agent_cfg(args.cfg)
    
    dm = DMConfig(
        config_file=args.game_config,
        wad=args.wad, # Adicionado
        map_name=args.map, # Adicionado
        total_players=args.players,
        port=args.port, 
        join_ip=args.ip,
        timelimit_minutes=args.timelimit, 
        render=args.render,
        stack_frames=args.stack
    )

    os.makedirs(agent.model_dir, exist_ok=True)
    save_path = os.path.join(agent.model_dir, agent.model_name)

    def _env():
        return DoomDMEnv(name=agent.name, is_host=args.host, dm=dm, agent_config=agent)

    # Se estiver rodando como script principal, inicia treino local
    print("--- INICIANDO TREINO LOCAL (STANDALONE) ---")
    mp.set_start_method("spawn", force=True)
    train_or_play(_env, dm.stack_frames, agent, save_path)
#!/usr/bin/env python3
import argparse, yaml, os, multiprocessing as mp
from .config import DMConfig, AgentConfig, EngineRewardConfig, ShapingConfig, PolicyConfig
from .env import DoomDMEnv
from .train import train_or_play

def load_agent_cfg(yaml_path: str) -> AgentConfig:
    with open(yaml_path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)
    er = EngineRewardConfig(**y.get("engine_reward", {}))
    sh = ShapingConfig(**y.get("shaping", {}))
    pol = PolicyConfig(**y.get("policy", {}))
    return AgentConfig(
        name=y.get("name", "Client"),
        colorset=y.get("colorset", 3),
        engine_reward=er,
        shaping=sh,
        policy=pol,
        model_dir=y.get("model_dir", "models"),
        model_name=y.get("model_name", "agent.zip"),
        train=bool(y.get("train", False)),
        train_steps=int(y.get("train_steps", 300_000)),
    )

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, help="YAML do agente")
    ap.add_argument("--port", type=int, default=5029)
    ap.add_argument("--ip", default="127.0.0.1")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--timelimit", type=float, default=3.0)
    ap.add_argument("--stack", type=int, default=4)
    args = ap.parse_args()

    agent = load_agent_cfg(args.cfg)
    dm = DMConfig(
        total_players=2,  # deve bater com o host
        port=args.port, join_ip=args.ip,
        timelimit_minutes=args.timelimit, render=args.render,
        stack_frames=args.stack
    )

    os.makedirs(agent.model_dir, exist_ok=True)
    save_path = os.path.join(agent.model_dir, agent.model_name)

    def _env():
        return DoomDMEnv(name=agent.name, is_host=False, dm=dm, agent=agent)

    mp.set_start_method("spawn", force=True)
    train_or_play(_env, dm.stack_frames, agent, save_path)

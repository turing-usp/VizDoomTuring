#!/usr/bin/env python3
import argparse, multiprocessing as mp
import numpy as np

from .config import DMConfig, AgentConfig
from .env import DoomDMEnv
from .train import make_vec_env

def host_proc(dm: DMConfig, agent: AgentConfig):
    def _env():
        return DoomDMEnv(name=agent.name, is_host=True, dm=dm, agent=agent)
    env = make_vec_env(_env, dm.stack_frames)
    obs = env.reset()
    # Mantém a instância viva; ação aleatória só para "tickar"
    while True:
        # VecEnv espera vetor de ações (n_envs=1) -> shape (1,)
        action = np.array([env.action_space.sample()], dtype=np.int64)
        obs, _r, dones, _ = env.step(action)
        if dones[0]:
            obs = env.reset()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--players", type=int, default=2)
    ap.add_argument("--port", type=int, default=5029)
    ap.add_argument("--timelimit", type=float, default=3.0)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--name", default="Host")
    args = ap.parse_args()

    dm = DMConfig(total_players=args.players, port=args.port,
                  timelimit_minutes=args.timelimit, render=args.render)
    agent = AgentConfig(name=args.name, train=False)

    mp.set_start_method("spawn", force=True)
    p = mp.Process(target=host_proc, args=(dm, agent), daemon=False)
    p.start(); p.join()

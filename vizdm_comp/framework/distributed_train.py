import argparse
import multiprocessing
import os
import pickle
import struct
import subprocess
import sys
import time
import gymnasium as gym
import numpy as np
import stable_baselines3
import torch
import yaml
from gymnasium import spaces
from multiprocessing.connection import Listener, Client, Connection
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv, VecTransposeImage, VecFrameStack
from typing import List, Tuple, Any, Optional, Dict, Union, Type

from .config import AgentConfig
from .client import load_agent_cfg

# ==============================================================================
# Helper para ler argumentos da Policy de forma segura
# ==============================================================================
def get_policy_args(policy_cfg) -> Dict[str, Any]:
    """
    Recupera argumentos da política tentando todas as variações possíveis.
    """
    # 1. Tenta .learn_kwargs (Onde você colocou no YAML)
    if hasattr(policy_cfg, "learn_kwargs") and isinstance(policy_cfg.learn_kwargs, dict):
        return policy_cfg.learn_kwargs

    # 2. Tenta .kwargs
    if hasattr(policy_cfg, "kwargs") and isinstance(policy_cfg.kwargs, dict):
        return policy_cfg.kwargs
    
    # 3. Tenta .args
    if hasattr(policy_cfg, "args") and isinstance(policy_cfg.args, dict):
        return policy_cfg.args
        
    # 4. Tenta .policy_kwargs (Às vezes o n_steps cai aqui por engano)
    if hasattr(policy_cfg, "policy_kwargs") and isinstance(policy_cfg.policy_kwargs, dict):
        return policy_cfg.policy_kwargs

    # 5. Fallback genérico
    if hasattr(policy_cfg, "__dict__"):
        return {k: v for k, v in vars(policy_cfg).items() if not k.startswith("_")}
        
    return {}

# ==============================================================================
# 1. Classe RemoteDMVecEnv
# ==============================================================================
class RemoteDMVecEnv(VecEnv):
    def __init__(self, conns: List[Connection], obs_space: spaces.Space, action_space: spaces.Space):
        self.conns = conns
        self.num_actors = len(conns)
        self.timeout = 600.0  # 10 minutos de tolerância

        super().__init__(self.num_actors, obs_space, action_space)
        self.actions: np.ndarray = None

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions
        for i, conn in enumerate(self.conns):
            act = self.actions[i]
            if isinstance(act, (np.integer, np.int64, np.int32)):
                act = int(act)
            try:
                conn.send({"cmd": "step", "action": act})
            except (BrokenPipeError, ConnectionResetError):
                pass 

    def step_wait(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        results = self._wait_for_replies("step")
        obs_list, rew_list, done_list, info_list = [], [], [], []
        
        for idx, res in enumerate(results):
            if "error" in res:
                raise RuntimeError(f"[TRAIN] Erro no step do ator {idx}: {res['error']}")
            
            obs_list.append(res["obs"])
            rew_list.append(res["reward"])
            done_list.append(res["done"])
            info_list.append(res["info"])

            if res["done"]:
                try:
                    self.conns[idx].send({"cmd": "reset"})
                    reset_res = self._wait_single_reply(idx, "reset_pos_done")
                    if "error" in reset_res:
                        raise RuntimeError(f"[TRAIN] Erro no reset automático do ator {idx}: {reset_res['error']}")
                    
                    obs_list[-1] = reset_res["obs"]
                    info_list[-1]["terminal_observation"] = res["obs"]
                    info_list[-1].update(reset_res["info"])
                except Exception as e:
                    raise RuntimeError(f"[TRAIN] Erro no reset automático do ator {idx}: {e!r}")

        return np.stack(obs_list), np.array(rew_list), np.array(done_list), info_list

    def reset(self) -> np.ndarray:
        for conn in self.conns:
            try:
                conn.send({"cmd": "reset"})
            except:
                pass
        
        results = self._wait_for_replies("reset")
        obs_list = []
        for res in results:
            if "error" in res:
                raise RuntimeError(f"[TRAIN] Erro no reset: {res['error']}")
            obs_list.append(res["obs"])
        return np.stack(obs_list)

    def close(self) -> None:
        for conn in self.conns:
            try:
                conn.send({"cmd": "close"})
            except:
                pass

    def get_attr(self, attr_name: str, indices=None) -> List[Any]:
        if attr_name == "render_mode": return [None] * self.num_envs
        return [None] * self.num_envs

    def set_attr(self, attr_name: str, value: Any, indices=None) -> None: pass
    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs) -> List[Any]: return [None] * self.num_envs
    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices=None) -> List[bool]: return [False] * self.num_envs

    def _wait_for_replies(self, context: str) -> List[Any]:
        start = time.time()
        results = [None] * self.num_actors
        pending = set(range(self.num_actors))

        while pending:
            if time.time() - start > self.timeout:
                raise TimeoutError(f"[TRAIN] TIMEOUT CRÍTICO ({self.timeout}s) em {context}. Atores pendentes: {list(pending)}.")

            active_conns = []
            for i in pending:
                if self.conns[i].poll(0.01): active_conns.append(i)
            
            for i in active_conns:
                try:
                    msg = self.conns[i].recv()
                    results[i] = msg
                    pending.remove(i)
                except (EOFError, ConnectionResetError):
                    raise RuntimeError(f"[TRAIN] Ator {i} desconectou inesperadamente em {context}.")

            if pending: time.sleep(0.005)

        return results

    def _wait_single_reply(self, idx: int, context: str) -> Any:
        start = time.time()
        conn = self.conns[idx]
        while True:
            if time.time() - start > self.timeout:
                raise TimeoutError(f"[TRAIN] TIMEOUT CRÍTICO ({self.timeout}s) em {context}({idx}).")
            if conn.poll(0.1):
                try: return conn.recv()
                except Exception as e: raise RuntimeError(f"[TRAIN] Erro ao receber de {idx}: {e}")

# ==============================================================================
# 2. Funções Auxiliares (Model, Config)
# ==============================================================================

def auto_adjust_n_steps(agent_cfg: AgentConfig, env: VecEnv, max_rollout_gib: float = 4.0, safety_factor: float = 4.0) -> AgentConfig:
    """
    Calcula n_steps. SE NÃO ENCONTRAR NO YAML, FORÇA 128 (SEGURO).
    """
    pol_args = get_policy_args(agent_cfg.policy)
    
    # 1. Se achou no YAML, usa e fica feliz.
    if "n_steps" in pol_args:
        print(f"[TRAIN] Usando n_steps do YAML: {pol_args['n_steps']}")
        return agent_cfg

    # 2. Se não achou, vamos calcular, mas com um teto seguro.
    obs_shape = env.observation_space.shape
    pixel_bytes = 1 
    obs_size = np.prod(obs_shape) * pixel_bytes
    num_envs = env.num_envs
    
    # Calculo de memória (apenas informativo agora)
    limit_bytes = (max_rollout_gib * 1024**3) / safety_factor
    max_steps_ram = int(limit_bytes / (num_envs * obs_size))
    
    # --- MUDANÇA CRÍTICA AQUI ---
    # Antes ele tentava chegar até 4096. Agora vamos ser conservadores.
    # Se o usuário não configurou, assumimos que ele quer performance/fluidez.
    print(f"[TRAIN][WARN] 'n_steps' não encontrado no YAML. RAM permite {max_steps_ram}, mas forçando modo SAFE.")
    
    chosen = 128 # Força 128 se não especificado. Chega de 4096.
    
    print(f"[TRAIN] Auto-ajuste n_steps: Escolhido={chosen} (Safe Mode).")
    
    # Injeta no dicionário para o SB3 usar
    pol_args["n_steps"] = chosen
    return agent_cfg

def build_model(agent_cfg: AgentConfig, env: VecEnv, save_path: str) -> PPO:
    # --- CORREÇÃO DO BUG ".ZIP DUPLO" ---
    # Se o caminho já tem .zip, usa ele. Se não, adiciona.
    file_to_check = save_path if save_path.endswith(".zip") else f"{save_path}.zip"
    
    if os.path.exists(file_to_check):
        print(f"[TRAIN] SUCESSO! Carregando modelo existente: {file_to_check}")
        # Carrega usando o caminho exato
        model = PPO.load(file_to_check, env=env, print_system_info=False)
    else:
        print(f"[TRAIN] Criando NOVO modelo PPO: {agent_cfg.model_name}")
        
        raw_args = get_policy_args(agent_cfg.policy)
        policy_kwargs = raw_args.copy() if raw_args else {}
        
        # Pega os valores
        learning_rate = float(policy_kwargs.pop("learning_rate", 1e-4))
        n_steps = int(policy_kwargs.pop("n_steps", 128))
        batch_size = int(policy_kwargs.pop("batch_size", 64))
        n_epochs = int(policy_kwargs.pop("n_epochs", 3))
        gamma = float(policy_kwargs.pop("gamma", 0.99))
        gae_lambda = float(policy_kwargs.pop("gae_lambda", 0.95))
        clip_range = float(policy_kwargs.pop("clip_range", 0.2))
        ent_coef = float(policy_kwargs.pop("ent_coef", 0.0))
        vf_coef = float(policy_kwargs.pop("vf_coef", 0.5))
        max_grad_norm = float(policy_kwargs.pop("max_grad_norm", 0.5))

        model_kwargs = {}
        if "net_arch" in policy_kwargs:
             model_kwargs["net_arch"] = policy_kwargs.pop("net_arch")
        if "activation_fn" in policy_kwargs:
             act_fn_name = policy_kwargs.pop("activation_fn")
             if act_fn_name == "tanh": model_kwargs["activation_fn"] = torch.nn.Tanh
             elif act_fn_name == "relu": model_kwargs["activation_fn"] = torch.nn.ReLU

        print(f"[TRAIN] Config Final PPO: n_steps={n_steps}, batch_size={batch_size}, n_epochs={n_epochs}")

        model = PPO(
            "CnnPolicy",
            env,
            verbose=1,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            policy_kwargs=model_kwargs,
            tensorboard_log=os.path.join(agent_cfg.model_dir, "tb_logs"),
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
    
    return model

class DebugCallback(BaseCallback):
    def __init__(self, log_every: int = 1000, reward_window: int = 10000, verbose=0):
        super().__init__(verbose)
        self.log_every = log_every
        self.reward_window = reward_window
        self.ep_rewards = []
        self.ep_lengths = []

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])
        for idx, done in enumerate(dones):
            if done:
                info = infos[idx]
                if "episode" in info:
                    self.ep_rewards.append(info["episode"]["r"])
                    self.ep_lengths.append(info["episode"]["l"])

        if self.n_calls % self.log_every == 0:
            if len(self.ep_rewards) > self.reward_window:
                self.ep_rewards = self.ep_rewards[-self.reward_window:]
            mean_r = np.mean(self.ep_rewards) if self.ep_rewards else 0.0
            print(f"[DEBUG] num_timesteps={self.num_timesteps}, mean_reward_window={mean_r:.3f}")
        return True
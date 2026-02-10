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
    Tenta recuperar os argumentos da política (learning_rate, n_steps, etc)
    lidando com diferentes nomes de atributos (kwargs, args) ou dicionários.
    """
    # 1. Tenta .kwargs (Padrão novo)
    if hasattr(policy_cfg, "kwargs") and isinstance(policy_cfg.kwargs, dict):
        return policy_cfg.kwargs
    
    # 2. Tenta .args (Padrão antigo)
    if hasattr(policy_cfg, "args") and isinstance(policy_cfg.args, dict):
        return policy_cfg.args
        
    # 3. Se for um objeto genérico, tenta pegar __dict__ filtrando dunders
    if hasattr(policy_cfg, "__dict__"):
        return {k: v for k, v in vars(policy_cfg).items() if not k.startswith("_")}
        
    return {}

# ==============================================================================
# 1. Classe RemoteDMVecEnv (Onde o Timeout acontece)
# ==============================================================================
class RemoteDMVecEnv(VecEnv):
    """
    VecEnv que controla atores remotos via multiprocessing.Connection (TCP).
    """

    def __init__(self, conns: List[Connection], obs_space: spaces.Space, action_space: spaces.Space):
        self.conns = conns
        self.num_actors = len(conns)
        
        # Timeout aumentado para aguentar pausas longas de treino (10 minutos)
        self.timeout = 600.0  

        super().__init__(self.num_actors, obs_space, action_space)
        self.actions: np.ndarray = None

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions
        for i, conn in enumerate(self.conns):
            # Envia a ação para cada ator
            # Se for int, numpy int64 etc, converte pra int Python nativo pra garantir serialização
            act = self.actions[i]
            if isinstance(act, (np.integer, np.int64, np.int32)):
                act = int(act)
            
            try:
                conn.send({"cmd": "step", "action": act})
            except (BrokenPipeError, ConnectionResetError):
                pass 

    def step_wait(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        # Espera resposta de todos
        results = self._wait_for_replies("step")
        
        obs_list, rew_list, done_list, info_list = [], [], [], []
        
        for idx, res in enumerate(results):
            if "error" in res:
                # Se o ator reportou erro, tentamos resetar ou falhamos.
                # Aqui vamos levantar erro para parar o treino e debuggar.
                raise RuntimeError(f"[TRAIN] Erro no step do ator {idx}: {res['error']}")
            
            obs_list.append(res["obs"])
            rew_list.append(res["reward"])
            done_list.append(res["done"])
            info_list.append(res["info"])

            # Auto-reset se done=True (padrão SB3 VecEnv)
            if res["done"]:
                # Manda ordem de reset imediata para esse ator
                try:
                    self.conns[idx].send({"cmd": "reset"})
                    # Espera a resposta do reset IMEDIATAMENTE (sincrono para esse ator)
                    reset_res = self._wait_single_reply(idx, "reset_pos_done")
                    if "error" in reset_res:
                        raise RuntimeError(f"[TRAIN] Erro no reset automático do ator {idx}: {reset_res['error']}")
                    
                    # Atualiza a observação para a nova (do reset)
                    # O reward e done continuam sendo do passo que finalizou
                    obs_list[-1] = reset_res["obs"]
                    # Infos do reset geralmente sobrescrevem ou mergeiam, mas SB3 usa info do step final
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

    # --- IMPLEMENTAÇÃO DOS MÉTODOS ABSTRATOS DO VECENV ---
    def get_attr(self, attr_name: str, indices=None) -> List[Any]:
        if attr_name == "render_mode":
            return [None] * self.num_envs
        return [None] * self.num_envs

    def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
        pass

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs) -> List[Any]:
        return [None] * self.num_envs

    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices=None) -> List[bool]:
        return [False] * self.num_envs
    # -----------------------------------------------------

    def _wait_for_replies(self, context: str) -> List[Any]:
        start = time.time()
        results = [None] * self.num_actors
        pending = set(range(self.num_actors))

        while pending:
            if time.time() - start > self.timeout:
                raise TimeoutError(
                    f"[TRAIN] TIMEOUT CRÍTICO ({self.timeout}s) em {context}. "
                    f"Atores que não responderam: {list(pending)}."
                )

            active_conns = []
            for i in pending:
                if self.conns[i].poll(0.01): 
                    active_conns.append(i)
            
            for i in active_conns:
                try:
                    msg = self.conns[i].recv()
                    results[i] = msg
                    pending.remove(i)
                except (EOFError, ConnectionResetError):
                    raise RuntimeError(f"[TRAIN] Ator {i} desconectou inesperadamente em {context}.")

            if pending:
                time.sleep(0.005)

        return results

    def _wait_single_reply(self, idx: int, context: str) -> Any:
        start = time.time()
        conn = self.conns[idx]
        while True:
            if time.time() - start > self.timeout:
                raise TimeoutError(
                    f"[TRAIN] TIMEOUT CRÍTICO ({self.timeout}s) em {context}({idx})."
                )
            
            if conn.poll(0.1):
                try:
                    return conn.recv()
                except Exception as e:
                    raise RuntimeError(f"[TRAIN] Erro ao receber de {idx}: {e}")

# ==============================================================================
# 2. Funções Auxiliares (Model, Config)
# ==============================================================================

def auto_adjust_n_steps(agent_cfg: AgentConfig, env: VecEnv, max_rollout_gib: float = 4.0, safety_factor: float = 4.0) -> AgentConfig:
    """
    Calcula n_steps para não estourar a RAM.
    """
    # --- FIX: Usa get_policy_args para ser agnóstico a kwargs/args ---
    pol_args = get_policy_args(agent_cfg.policy)
    
    if "n_steps" in pol_args:
        print(f"[TRAIN] Usando n_steps do YAML: {pol_args['n_steps']}")
        return agent_cfg

    obs_shape = env.observation_space.shape
    pixel_bytes = 1 # uint8
    obs_size = np.prod(obs_shape) * pixel_bytes
    
    num_envs = env.num_envs
    limit_bytes = (max_rollout_gib * 1024**3) / safety_factor
    max_steps = int(limit_bytes / (num_envs * obs_size))
    
    suggestions = [128, 256, 512, 1024, 2048, 4096]
    chosen = 128
    for s in suggestions:
        if s <= max_steps:
            chosen = s
        else:
            break
            
    print(f"[TRAIN] Auto-ajuste n_steps: obs={obs_shape}, num_envs={num_envs}. Calculado max={max_steps}. Escolhido={chosen}.")
    
    # Injeta no dicionário correto
    # Se pol_args for referência direta ao dict dentro do objeto, isso atualiza o objeto
    pol_args["n_steps"] = chosen
    return agent_cfg

def build_model(agent_cfg: AgentConfig, env: VecEnv, save_path: str) -> PPO:
    """
    Carrega ou cria PPO.
    """
    if os.path.exists(save_path + ".zip"):
        print(f"[TRAIN] Carregando modelo existente: {save_path}.zip")
        model = PPO.load(save_path, env=env, print_system_info=True)
    else:
        print(f"[TRAIN] Criando NOVO modelo PPO: {agent_cfg.model_name}")
        
        # --- FIX: Usa get_policy_args para criar cópia dos argumentos ---
        raw_args = get_policy_args(agent_cfg.policy)
        policy_kwargs = raw_args.copy() if raw_args else {}
        
        learning_rate = float(policy_kwargs.pop("learning_rate", 1e-4))
        n_steps = int(policy_kwargs.pop("n_steps", 2048))
        batch_size = int(policy_kwargs.pop("batch_size", 64))
        n_epochs = int(policy_kwargs.pop("n_epochs", 10))
        gamma = float(policy_kwargs.pop("gamma", 0.99))
        gae_lambda = float(policy_kwargs.pop("gae_lambda", 0.95))
        clip_range = float(policy_kwargs.pop("clip_range", 0.2))
        ent_coef = float(policy_kwargs.pop("ent_coef", 0.0))
        vf_coef = float(policy_kwargs.pop("vf_coef", 0.5))
        max_grad_norm = float(policy_kwargs.pop("max_grad_norm", 0.5))

        # Separate net_arch if present
        model_kwargs = {}
        if "net_arch" in policy_kwargs:
             model_kwargs["net_arch"] = policy_kwargs.pop("net_arch")
        if "activation_fn" in policy_kwargs:
             act_fn_name = policy_kwargs.pop("activation_fn")
             if act_fn_name == "tanh":
                 model_kwargs["activation_fn"] = torch.nn.Tanh
             elif act_fn_name == "relu":
                 model_kwargs["activation_fn"] = torch.nn.ReLU

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
                    r = info["episode"]["r"]
                    l = info["episode"]["l"]
                    self.ep_rewards.append(r)
                    self.ep_lengths.append(l)

        if self.n_calls % self.log_every == 0:
            if len(self.ep_rewards) > self.reward_window:
                self.ep_rewards = self.ep_rewards[-self.reward_window:]
            
            mean_r = np.mean(self.ep_rewards) if self.ep_rewards else 0.0
            print(f"[DEBUG] num_timesteps={self.num_timesteps}, mean_reward_window={mean_r:.3f} (últimos {len(self.ep_rewards)} eps)")
            
        return True
from typing import Callable
import os

from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecFrameStack,
)
from gymnasium import spaces

from .policies import resolve_effective_algo, build_sb3, ExternalPolicyAdapter
from .runtime_cuda import configure_cuda_runtime


def make_vec_env(env_fn: Callable, n_stack: int):
    """
    Cria VecEnv de 1 ambiente (DummyVecEnv) + frame stack.
    """
    env = DummyVecEnv([env_fn])
    channels_order = {"image": "first", "state": None} if isinstance(env.observation_space, spaces.Dict) else "first"
    env = VecFrameStack(env, n_stack=n_stack, channels_order=channels_order)
    return env


def _load_or_create_sb3(algo_cls, env, agent_cfg, save_path: str):
    """
    Se existir modelo salvo, carrega. Caso contrário, cria novo via build_sb3.
    """
    if os.path.exists(save_path):
        try:
            print(f"[CLIENT] Carregando modelo existente: {save_path}")
            return algo_cls.load(save_path, env=env)
        except Exception as e:
            print(f"[CLIENT][WARN] Falha ao carregar modelo: {e!r}. Criando modelo novo.")

    print(f"[CLIENT] Criando novo modelo SB3 ({agent_cfg.policy.algo}).")
    return build_sb3(
        algo_cls,
        "CnnPolicy",
        env,
        agent_cfg.policy.policy_kwargs,
        agent_cfg.policy.learn_kwargs,
    )


def train_or_play(env_fn: Callable, n_stack: int, agent_cfg, save_path: str):
    """
    Se agent_cfg.policy.algo == external: roda adapter .predict (sem treino).
    Caso contrário, instancia SB3 e treina/infere conforme config.

    - Se agent_cfg.train == True: treina em chunks e salva.
    - Se agent_cfg.train == False: apenas joga usando o modelo (carregado de save_path se existir).
    """
    env = make_vec_env(env_fn, n_stack)
    algo_name = agent_cfg.policy.algo.lower()

    # ------------------------------------------------------------------
    # Caso especial: política externa (PyTorch puro, sem SB3)
    # ------------------------------------------------------------------
    if algo_name == "external":
        adapter = ExternalPolicyAdapter(
            weights_path=agent_cfg.policy.external_path,
            n_actions=env.action_space.n,
            external_class=agent_cfg.policy.policy_kwargs.get("external_class"),
        )
        obs = env.reset()
        while True:
            action, _ = adapter.predict(obs)
            obs, _rew, dones, _infos = env.step(action)
            if dones[0]:
                obs = env.reset()
        # sem return

    # ------------------------------------------------------------------
    # Caso SB3 (PPO / A2C / DQN)
    # ------------------------------------------------------------------
    else:
        configure_cuda_runtime(
            benchmark=bool(agent_cfg.policy.learn_kwargs.get("use_cudnn_benchmark", True)),
            tf32=bool(agent_cfg.policy.learn_kwargs.get("use_tf32", False)),
            matmul_precision=str(agent_cfg.policy.learn_kwargs.get("torch_matmul_precision", "highest")),
        )
        algo_cls = resolve_effective_algo(algo_name, agent_cfg.policy.learn_kwargs)
        model = _load_or_create_sb3(algo_cls, env, agent_cfg, save_path)

        if agent_cfg.train:
            # Modo treino
            remaining = int(agent_cfg.train_steps)
            chunk = max(10_000, remaining // 10)
            while remaining > 0:
                cur = min(chunk, remaining)
                print(f"[CLIENT][TRAIN] learn(total_timesteps={cur}), remaining={remaining}")
                model.learn(total_timesteps=cur, reset_num_timesteps=False)
                model.save(save_path)
                remaining -= cur
            print(f"[CLIENT][TRAIN] Treino concluído. Modelo salvo em {save_path}.")
        else:
            # Modo "play" (inferência somente)
            print(f"[CLIENT][PLAY] Iniciando jogo com modelo {save_path}.")
            obs = env.reset()
            while True:
                # deterministic=True para comportamento mais estável em avaliação
                action, _ = model.predict(obs, deterministic=True)
                obs, _rew, dones, _infos = env.step(action)
                if dones[0]:
                    obs = env.reset()

        # Garante salvar a última versão mesmo em play (opcional, mas seguro)
        model.save(save_path)

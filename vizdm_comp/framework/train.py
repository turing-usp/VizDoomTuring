from typing import Callable
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack
from .policies import resolve_algo, build_sb3, ExternalPolicyAdapter

def make_vec_env(env_fn: Callable, n_stack: int):
    env = DummyVecEnv([env_fn])
    env = VecTransposeImage(env)                 # HWC->CHW
    env = VecFrameStack(env, n_stack=n_stack, channels_order="first")
    return env

def train_or_play(env_fn: Callable, n_stack: int, agent_cfg, save_path: str):
    """
    Se agent_cfg.policy.algo == external: roda adapter .predict (sem treino).
    Caso contrário, instancia SB3 e treina/infere conforme config.
    """
    env = make_vec_env(env_fn, n_stack)
    algo_name = agent_cfg.policy.algo.lower()

    if algo_name == "external":
        adapter = ExternalPolicyAdapter(
            weights_path=agent_cfg.policy.external_path,
            n_actions=env.action_space.n,
            external_class=agent_cfg.policy.policy_kwargs.get("external_class")
        )
        obs = env.reset()
        while True:
            action, _ = adapter.predict(obs)
            obs, _rew, dones, _infos = env.step(action)
            if dones[0]:
                obs = env.reset()
        # sem return
    else:
        algo_cls = resolve_algo(algo_name)
        model = build_sb3(algo_cls, "CnnPolicy", env,
                          agent_cfg.policy.policy_kwargs, agent_cfg.policy.learn_kwargs)
        if agent_cfg.train:
            remaining = int(agent_cfg.train_steps)
            chunk = max(10_000, remaining // 10)
            while remaining > 0:
                cur = min(chunk, remaining)
                model.learn(total_timesteps=cur, reset_num_timesteps=False)
                model.save(save_path)
                remaining -= cur
        else:
            obs = env.reset()
            while True:
                action, _ = model.predict(obs, deterministic=False)
                obs, _rew, dones, _infos = env.step(action)
                if dones[0]:
                    obs = env.reset()
        model.save(save_path)

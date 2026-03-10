from typing import Callable
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack
from .policies import resolve_algo, build_sb3, ExternalPolicyAdapter
import os



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
        
        # --- A MÁGICA DO CARREGAMENTO AQUI ---
        if os.path.exists(save_path):
            print(f"[TRAIN] SUCESSO! Carregando cérebro existente: {save_path}")
            # Carrega os pesos salvos e conecta ao ambiente atual
            model = algo_cls.load(save_path, env=env)
        else:
            print(f"[TRAIN] Arquivo não encontrado. Criando NOVO modelo PPO: {save_path}")
            # Só cria um zero bala se o arquivo não existir
            model = build_sb3(algo_cls, "CnnPolicy", env,
                              agent_cfg.policy.policy_kwargs, agent_cfg.policy.learn_kwargs)
        # -------------------------------------

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

#ALTERAÇÃO que deve ser feita dps de ter os modelos carregados
# else:
#             # --- MODO JOGO (PLAY) COM TROCA DE CÉREBRO ---
#             print("--- INICIANDO MODO DE JOGO DINÂMICO ---")
            
#             # 1. Carrega os dois cérebros (assumindo que já foram treinados)
#             # Ajuste os caminhos se necessário
#             path_pegador = os.path.join("models", "pegador_bot.zip")
#             path_fugitivo = os.path.join("models", "fugitivo_bot.zip")
            
#             # Verifica se existem
#             if not os.path.exists(path_pegador) or not os.path.exists(path_fugitivo):
#                 print("ERRO: É necessário ter treinado 'pegador_bot.zip' e 'fugitivo_bot.zip' antes!")
#                 return

#             print("Carregando cérebro do Pegador...")
#             brain_pegador = algo_cls.load(path_pegador)
#             print("Carregando cérebro do Fugitivo...")
#             brain_fugitivo = algo_cls.load(path_fugitivo)

#             # 2. Define quem eu sou agora com base no nome do agente no YAML
#             current_role = "PEGADOR" if "Pegador" in agent_cfg.name else "FUGITIVO"
#             current_brain = brain_pegador if current_role == "PEGADOR" else brain_fugitivo
            
#             print(f"Começando como: {current_role}")

#             obs = env.reset()
#             while True:
#                 # Usa o cérebro atual para decidir
#                 action, _ = current_brain.predict(obs, deterministic=False)
                
#                 # Executa a ação
#                 obs, rewards, dones, _infos = env.step(action)
                
#                 # --- LÓGICA DE TROCA DE CÉREBRO ---
#                 # O ambiente vetorizado retorna um array de rewards, pegamos o [0]
#                 r = rewards[0]

#                 # Regra 1: Sou Pegador e toquei em alguém (+10 pontos)
#                 if current_role == "PEGADOR" and r >= 5.0: # Usei >5 para margem de segurança
#                     print(f"PEGUEI! (Reward: {r}) -> Virando Fugitivo!")
#                     current_role = "FUGITIVO"
#                     current_brain = brain_fugitivo
                
#                 # Regra 2: Sou Fugitivo e fui tocado (-10 pontos)
#                 elif current_role == "FUGITIVO" and r <= -5.0:
#                     print(f"FUI PEGO! (Reward: {r}) -> Virando Pegador!")
#                     current_role = "PEGADOR"
#                     current_brain = brain_pegador

#                 if dones[0]:
#                     obs = env.reset()
#                     # Opcional: Você quer resetar o papel quando morre/reinicia?
#                     # Se não, ele respawna com o último papel que tinha.
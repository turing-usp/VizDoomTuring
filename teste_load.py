import gymnasium.spaces
import numpy as np
from stable_baselines3 import PPO

MODEL_PATH = "models/pegador_bot.zip"

# Cria espaço falso só para carregar
dummy_obs = gymnasium.spaces.Box(low=0, high=255, shape=(120, 160, 1), dtype=np.uint8)
dummy_act = gymnasium.spaces.Discrete(8)

try:
    model = PPO.load(
        MODEL_PATH,
        custom_objects={
            "observation_space": dummy_obs,
            "action_space": dummy_act
        },
        map_location="cpu"
    )
    print(f"--- VERDADE DO ARQUIVO ---")
    print(f"O arquivo foi salvo com n_steps: {model.n_steps}")
    print(f"O arquivo foi salvo com batch_size: {model.batch_size}")
    print(f"--------------------------")
except Exception as e:
    print(f"Erro ao ler: {e}")
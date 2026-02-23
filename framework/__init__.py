from .config import (
    DMConfig, EngineRewardConfig, ShapingConfig, PolicyConfig, AgentConfig
)
from .env import DoomDMEnv
#python run_5_train_shared.py --cfg example_agent.yaml --num-matches 2 --actors-per-match 10 --render host
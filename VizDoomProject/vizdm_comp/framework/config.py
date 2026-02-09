from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Literal

AlgoName = Literal["ppo", "a2c", "dqn", "external"]
ExtractorName = Literal["cnn_default", "custom"]

@dataclass(frozen=True)
class DMConfig:
    """Config do servidor/jogo multiplayer."""
    config_file: str = "tag.cfg"
    total_players: int = 5
    port: int = 5029
    join_ip: str = "127.0.0.1"
    map_name: str = "map01"
    timelimit_minutes: float = 3.0
    render: bool = False
    frame_skip: int = 12
    screen_w: int = 84
    screen_h: int = 84
    stack_frames: int = 4

@dataclass(frozen=True)
class EngineRewardConfig:
    """
    Recompensas nativas do ViZDoom (entram no retorno de make_action()).
    Use zero para desligar um canal. Alguns campos requerem ViZDoom >= 1.3.0.
    """
    living_reward: float = 0.0
    frag_reward: float = 0.0
    hit_reward: float = 0.0
    hit_taken_reward: float = 0.0
    damage_made_reward: float = 0.0
    damage_taken_penalty: float = 0.0
    item_reward: float = 0.0
    health_reward: float = 0.0
    armor_reward: float = 0.0
    secret_reward: float = 0.0

@dataclass(frozen=True)
class ShapingConfig:
    """
    Recompensa própria do agente (ex.: deltas de variáveis).
    """
    w_frag: float = 1.0          # ΔFRAGCOUNT
    w_health: float = 0.0        # ΔHEALTH
    w_armor: float = 0.0         # ΔARMOR
    w_ammo2_cost: float = 0.0    # penaliza gasto de AMMO2
    w_hits: float = 0.0          # ΔHITCOUNT
    w_hits_taken: float = 0.0    # -ΔHITS_TAKEN (use negativo)
    w_deaths: float = 0.0        # -ΔDEATHCOUNT (use negativo)
    step_penalty: float = 0.0
    include_engine_reward: bool = False  # soma reward do engine?

@dataclass(frozen=True)
class PolicyConfig:
    """Escolha do algoritmo, extrator e hiperparâmetros."""
    algo: AlgoName = "ppo"
    extractor: ExtractorName = "cnn_default"
    external_path: Optional[str] = None   # .pth (quando algo == "external")
    policy_kwargs: Dict[str, Any] = field(default_factory=dict)
    learn_kwargs: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class AgentConfig:
    """
    Config por agente. Cada cliente lê o SEU YAML e treina/joga com essas regras.
    """
    name: str = "Agent"
    colorset: int = 3
    engine_reward: EngineRewardConfig = EngineRewardConfig()
    shaping: ShapingConfig = ShapingConfig()
    policy: PolicyConfig = PolicyConfig()
    model_dir: str = "models"
    model_name: str = "agent.zip"    # SB3
    train: bool = False
    train_steps: int = 300_000

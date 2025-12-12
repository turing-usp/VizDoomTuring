from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Literal

AlgoName = Literal["ppo", "a2c", "dqn", "external"]
ExtractorName = Literal["cnn_default", "custom"]


@dataclass(frozen=True)
class DMConfig:
    """Config do servidor/jogo multiplayer."""
    total_players: int = 5
    port: int = 5029
    join_ip: str = "127.0.0.1"
    map_name: str = "map01"
    timelimit_minutes: float = 3.0
    render: bool = False
    frame_skip: int = 4
    screen_w: int = 160
    screen_h: int = 120
    stack_frames: int = 4


@dataclass(frozen=True)
class EngineRewardConfig:
    """
    Recompensas nativas do ViZDoom.
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
    Recompensa auxiliar calculada no Python.
    include_engine_reward está obsoleto: o RewardShaper não soma mais o engine reward.
    """
    w_frag: float = 1.0
    w_health: float = 0.0
    w_armor: float = 0.0
    w_ammo2_cost: float = 0.0
    w_hits: float = 0.0
    w_hits_taken: float = 0.0
    w_deaths: float = 0.0
    step_penalty: float = 0.0
    include_engine_reward: bool = False


@dataclass(frozen=True)
class RenderSettingsConfig:
    """
    Configurações visuais do VizDoom.

    - A janela de renderização (tela que você vê) será configurada em alta resolução
      diretamente no env.py (ex.: RES_640X480).
    - 'resolution' define a resolução da OBSERVAÇÃO da REDE (o que vai para o modelo).
    """
    resolution: str = "RES_160X120"  # Resolução da REDE (ex.: RES_120X90, RES_160X120)
    format: str = "GRAY8"            # Formato da janela/render (RGB24 ou GRAY8)
    hud: bool = False                # Mostrar interface


@dataclass(frozen=True)
class PolicyConfig:
    algo: AlgoName = "ppo"
    extractor: ExtractorName = "cnn_default"
    external_path: Optional[str] = None
    policy_kwargs: Dict[str, Any] = field(default_factory=dict)
    learn_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RewardConfig:
    """
    Bloco unificado de recompensa:
    - engine: recompensas nativas do ViZDoom (living, frag, etc.)
    - shaping: recompensa de modelagem em Python (por deltas de variáveis)
    """
    engine: EngineRewardConfig = EngineRewardConfig()
    shaping: ShapingConfig = ShapingConfig()


@dataclass(frozen=True)
class AgentConfig:
    """
    Configuração completa do Agente.
    """
    name: str = "Agent"
    colorset: int = 3

    # Configuração de renderização da janela do VizDoom
    render_settings: RenderSettingsConfig = RenderSettingsConfig()

    # Bloco unificado de recompensa
    reward: RewardConfig = RewardConfig()

    policy: PolicyConfig = PolicyConfig()
    model_dir: str = "models"
    model_name: str = "agent.zip"
    train: bool = False
    train_steps: int = 300_000
    stack_frames: int = 4

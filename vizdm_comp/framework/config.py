from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Literal

AlgoName = Literal["ppo", "a2c", "dqn", "external"]
ExtractorName = Literal["cnn_default", "custom"]

@dataclass(frozen=True)
class RenderSettingsConfig:
    resolution: str = "RES_160X120"
    format: str = "CRCGCB"
    render_hud: bool = False
    render_crosshair: bool = False
    render_weapon: bool = True
    render_decals: bool = False
    render_particles: bool = False
    window_visible: bool = False

@dataclass(frozen=True)
class DMConfig:
    """Config do servidor/jogo multiplayer."""
    config_file: str = "tag.cfg"
    
    # --- CORREÇÃO DO ERRO ATUAL ---
    # Adicionamos o campo 'wad' para o ator conseguir salvar essa info
    wad: Optional[str] = "freedm.wad"
    # ------------------------------

    total_players: int = 5
    port: int = 5029
    join_ip: str = "127.0.0.1"
    map_name: str = "map01"
    timelimit_minutes: float = 3.0
    render: bool = False
    frame_skip: int = 4
    screen_w: int = 84
    screen_h: int = 84
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
    Recompensa própria do agente.
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
class PolicyConfig:
    """Escolha do algoritmo, extrator e hiperparâmetros."""
    algo: AlgoName = "ppo"
    extractor: ExtractorName = "cnn_default"
    external_path: Optional[str] = None   
    policy_kwargs: Dict[str, Any] = field(default_factory=dict)
    learn_kwargs: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class AgentConfig:
    """
    Config por agente.
    """
    name: str = "Agent"
    colorset: int = 3
    
    # Render settings vindo do YAML
    render_settings: RenderSettingsConfig = field(default_factory=RenderSettingsConfig)
    
    engine_reward: EngineRewardConfig = field(default_factory=EngineRewardConfig)
    shaping: ShapingConfig = field(default_factory=ShapingConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    
    model_dir: str = "models"
    model_name: str = "agent.zip"    
    train: bool = False
    train_steps: int = 300_000

@dataclass(frozen=True)
class RewardConfig:
    """Container para Engine e Shaping (necessário para distributed_actor)"""
    engine: EngineRewardConfig
    shaping: ShapingConfig
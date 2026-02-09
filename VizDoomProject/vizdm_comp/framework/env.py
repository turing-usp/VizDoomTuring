from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import vizdoom as vzd
from vizdoom import Mode, Button, GameVariable
import cv2
from .config import DMConfig, AgentConfig
from .rewards import RewardShaper, apply_engine_rewards

import os

# --- LISTAS DE AÇÕES GLOBAIS (ATUALIZADAS) ---

# Ações para o modo Deathmatch (sem alteração)
actions_deathmatch = np.asarray([
    [0,0,0,0,0,0], [0,0,1,0,0,0], [0,0,0,1,0,0], [0,0,0,0,1,0],
    [0,0,1,0,0,1], [1,0,0,0,0,0], [0,1,0,0,0,0], [0,0,0,0,0,1]
], dtype=np.int32)
buttons_deathmatch = [
    Button.MOVE_LEFT, Button.MOVE_RIGHT, Button.MOVE_FORWARD,
    Button.TURN_LEFT, Button.TURN_RIGHT, Button.ATTACK
]

# --- AÇÕES ATUALIZADAS PARA O MODO "PEGA-PEGA" ---
buttons_tag = [
    Button.MOVE_FORWARD,
    Button.MOVE_BACKWARD,
    Button.TURN_LEFT, 
    Button.TURN_RIGHT,
    Button.SPEED,
    Button.ATTACK  # <-- RE-ADICIONADO
]
actions_tag = np.asarray([
    # [FWD, BWD, T_LEFT, T_RIGHT, SPEED, ATTACK]
    [0,0,0,0,0,0], # 0: Nada
    [1,0,0,0,0,0], # 1: Mover Frente
    [1,0,0,0,1,0], # 2: Correr Frente
    [0,1,0,0,0,0], # 3: Mover Trás
    [0,0,1,0,0,0], # 4: Virar Esquerda
    [0,0,0,1,0,0], # 5: Virar Direita
    [1,0,0,0,0,1], # 6: Frente + Soco (Pega!)
    [1,0,0,0,1,1]  # 7: Correr + Soco (Pega!)
], dtype=np.int32)
# --- FIM DAS ATUALIZAÇÕES ---


class DoomDMEnv(gym.Env):
    """
    Env single-agent (host OU cliente) para deathmatch VizDoom.
    Reward = shaping (+ opcional reward nativo do engine).
    Observação HWC; wrappers externos fazem CHW/stack.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, name: str, is_host: bool, dm: DMConfig, agent: AgentConfig):
        super().__init__()
        self.name = name
        self.is_host = is_host
        self.dm = dm
        self.agent = agent
        g = self.game = vzd.DoomGame()
        g.load_config(dm.config_file)
        g.set_doom_map(self.dm.map_name)
        g.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
        g.set_screen_format(vzd.ScreenFormat.GRAY8)
        g.set_render_hud(False); g.set_render_crosshair(False)
        g.set_window_visible(self.dm.render)

        config_filename = os.path.basename(dm.config_file)

        if "tag" in config_filename:
            self.game_mode = "tag"
            available_buttons = buttons_tag
            self._actions = actions_tag
            print(f"--- MODO 'PEGA-PEGA' (TAG) CARREGADO PARA {name} ---")
            g.set_render_weapon(False)
        else:
            self.game_mode = "deathmatch"
            available_buttons = buttons_deathmatch
            self._actions = actions_deathmatch
            print(f"--- MODO 'DEATHMATCH' (PADRÃO) CARREGADO PARA {name} ---")
            g.set_render_weapon(True)
            
        g.set_available_buttons(available_buttons)

        # Variáveis para shaping/infos
        g.add_available_game_variable(GameVariable.HEALTH)
        g.add_available_game_variable(GameVariable.ARMOR)
        g.add_available_game_variable(GameVariable.AMMO2)
        g.add_available_game_variable(GameVariable.FRAGCOUNT)
        g.add_available_game_variable(GameVariable.DEATHCOUNT)
        g.add_available_game_variable(GameVariable.HITCOUNT)
        g.add_available_game_variable(GameVariable.HITS_TAKEN)

        apply_engine_rewards(g, self.agent.engine_reward)

        g.set_mode(Mode.ASYNC_PLAYER)  # multiplayer exige

        common = (
            f"-deathmatch +timelimit {self.dm.timelimit_minutes} "
            f"+sv_forcerespawn 1 +sv_noautoaim 1 +sv_respawnprotect 1 +sv_spawnfarthest 1 "
            f"-skill 3 +name {self.name} +colorset {self.agent.colorset} "
        )
        if self.is_host:
            args = f"-host {self.dm.total_players} -port {self.dm.port} +viz_connect_timeout 300 "
        else:
            args = f"-join {self.dm.join_ip} -port {self.dm.port} "
        g.add_game_args(args + common)
        g.init()


        self.shaper = RewardShaper(self.agent.shaping)
        self._last_obs: Optional[np.ndarray] = None

        H, W = self.dm.screen_h, self.dm.screen_w
        self.observation_space = spaces.Box(low=0, high=255, shape=(H, W, 1), dtype=np.uint8)
        self.action_space = spaces.Discrete(len(self._actions))

        self.shaper = RewardShaper(self.agent.shaping)
        self._last_obs: Optional[np.ndarray] = None

    def _read_obs(self):
        state = self.game.get_state()
        
        # Se o jogo acabou, retorna tela preta no tamanho correto (84x84)
        if state is None:
            return np.zeros(self.observation_space.shape, dtype=np.uint8)

        buf = state.screen_buffer
        
        # --- AQUI ESTÁ A MÁGICA: RESIZE ---
        # O ViZDoom entrega 160x120. Nós queremos o que está no config (84x84).
        # cv2.resize espera (Largura, Altura).
        target_w = self.dm.screen_w
        target_h = self.dm.screen_h
        
        # Redimensiona a imagem usando interpolação de área (melhor para encolher)
        buf = cv2.resize(buf, (target_w, target_h), interpolation=cv2.INTER_AREA)
        # ----------------------------------

        # Se for Grayscale (2D), adiciona a dimensão de canal no final (H, W, 1)
        if buf.ndim == 2:
            buf = buf[..., None]

        # Debug de segurança (opcional, pode remover depois)
        if buf.shape != self.observation_space.shape:
             # Se cair aqui, algo muito estranho aconteceu com o config
             print(f"ERRO DE SHAPE: Esperado {self.observation_space.shape}, Recebido {buf.shape}")

        return buf

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if not self.game.is_running() or self.game.is_episode_finished():
            try:
                self.game.new_episode()
            except Exception:
                import time
                for _ in range(600):
                    time.sleep(0.03)
                    if self.game.is_running() and not self.game.is_episode_finished():
                        break
        self.shaper.reset(self.game)
        obs = self._read_obs()
        return obs, {"name": self.name}

    def step(self, action: int):
        if self.game.is_player_dead():
            self.game.respawn_player()
        a = self._actions[action].tolist()
        engine_r = float(self.game.make_action(a, self.dm.frame_skip))  # reward do engine
        shaped_r = self.shaper.compute(self.game, engine_r)
        done = self.game.is_episode_finished()
        obs = self._read_obs()
        return obs, shaped_r, done, False, {"engine_r": engine_r}

    def render(self): pass

    def close(self):
        try: self.game.close()
        except: pass

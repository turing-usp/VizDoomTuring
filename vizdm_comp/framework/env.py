from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import vizdoom as vzd
from vizdoom import Mode, Button, GameVariable
import cv2
import os

from .config import DMConfig, AgentConfig
from .rewards import RewardShaper, apply_engine_rewards

import cv2
import numpy as np


# --- LISTAS DE AÇÕES ---
actions_deathmatch = np.asarray([
    [0,0,0,0,0,0], [0,0,1,0,0,0], [0,0,0,1,0,0], [0,0,0,0,1,0],
    [0,0,1,0,0,1], [1,0,0,0,0,0], [0,1,0,0,0,0], [0,0,0,0,0,1]
], dtype=np.int32)
buttons_deathmatch = [
    Button.MOVE_LEFT, Button.MOVE_RIGHT, Button.MOVE_FORWARD,
    Button.TURN_LEFT, Button.TURN_RIGHT, Button.ATTACK
]

buttons_tag = [
    Button.MOVE_FORWARD, Button.MOVE_BACKWARD,
    Button.TURN_LEFT, Button.TURN_RIGHT,
    Button.SPEED, Button.ATTACK 
]
actions_tag = np.asarray([
    [0,0,0,0,0,0], [1,0,0,0,0,0], [1,0,0,0,1,0], [0,1,0,0,0,0], 
    [0,0,1,0,0,0], [0,0,0,1,0,0], [1,0,0,0,0,1], [1,0,0,0,1,1]  
], dtype=np.int32)

class DoomDMEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, name: str, is_host: bool, dm: DMConfig, agent_config: AgentConfig):
        super().__init__()
        self.name = name
        self.is_host = is_host
        self.dm = dm
        self.agent = agent_config
        
        g = self.game = vzd.DoomGame()
        
        # Configuração de WAD
        if self.dm.wad:
            root_dir = os.getcwd()
            wad_candidate_root = os.path.join(root_dir, self.dm.wad)
            framework_dir = os.path.dirname(os.path.abspath(__file__))
            wad_candidate_fw = os.path.join(framework_dir, os.path.basename(self.dm.wad))

            if os.path.exists(wad_candidate_root):
                g.set_doom_game_path(wad_candidate_root)
            elif os.path.exists(wad_candidate_fw):
                g.set_doom_game_path(wad_candidate_fw)
            else:
                print(f"[ENV] AVISO: WAD {self.dm.wad} não encontrado.")

        g.set_doom_map(self.dm.map_name)
        # --- CONFIGURAÇÃO VISUAL (160x120, GRAY8) ---
        g.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
        g.set_screen_format(vzd.ScreenFormat.RGB24)
        g.set_render_hud(False)
        g.set_render_crosshair(False)
        g.set_window_visible(self.dm.render)

        # Detecta modo
        config_filename = str(dm.config_file).lower()
        if "tag" in config_filename:
            self.game_mode = "tag"
            available_buttons = buttons_tag
            self._actions = actions_tag
            g.set_render_weapon(False) 
        else:
            self.game_mode = "deathmatch"
            available_buttons = buttons_deathmatch
            self._actions = actions_deathmatch
            g.set_render_weapon(True)
            
        g.set_available_buttons(available_buttons)

        # Variáveis do Jogo
        g.add_available_game_variable(GameVariable.HEALTH)
        g.add_available_game_variable(GameVariable.ARMOR)
        g.add_available_game_variable(GameVariable.AMMO2)
        g.add_available_game_variable(GameVariable.FRAGCOUNT)
        g.add_available_game_variable(GameVariable.DEATHCOUNT)
        g.add_available_game_variable(GameVariable.HITCOUNT)
        g.add_available_game_variable(GameVariable.HITS_TAKEN)

        apply_engine_rewards(g, self.agent.engine_reward)
        g.set_mode(Mode.ASYNC_PLAYER) 

        # --- ARGS DO JOGO (COM PROTEÇÃO ANTI-ERRO) ---
        common = (
            f"-deathmatch +timelimit {self.dm.timelimit_minutes} "
            f"+sv_forcerespawn 1 +sv_noautoaim 1 +sv_respawnprotect 1 +sv_spawnfarthest 1 "
            f"-skill 3 +name {self.name} +colorset {self.agent.colorset} "
            f"-nosound -noconsole "
            f"+sv_noitems 1 "     # Sem armas no chão
            f"+give fist "        # Soco garantido
            f"+sv_nointermission 1 " # Pula placar (Evita timeout de 600s)
        )

        if self.is_host:
            args = f"-host {self.dm.total_players} -port {self.dm.port} +viz_connect_timeout 60 "
        else:
            args = f"-join {self.dm.join_ip} -port {self.dm.port} "
        g.add_game_args(args + common)
        
        import time
        if not self.is_host and self.dm.render:
            # Faz o Joiner esperar 2 segundos para a janela do Host abrir primeiro
            time.sleep(2.0) 
        # ---------------------------------------

        try:
            g.init()
            print(f"[ENV DEBUG] VizDoom INICIADO com sucesso para: {self.name}")
        except Exception as e:
            print(f"[ENV CRASH] Erro ao iniciar VizDoom: {e}")
            raise e

        self.shaper = RewardShaper(self.agent.shaping)
        self._last_obs: Optional[np.ndarray] = None

        # --- DEFINIÇÃO DO ESPAÇO DE OBSERVAÇÃO ---
        self.observation_space = spaces.Box(low=0, high=255, shape=(120, 160, 3), dtype=np.uint8)
        self.action_space = spaces.Discrete(len(self._actions))

        # =================================================================
        # [CÂMERA DE SEGURANÇA] -> AGORA SIM NO FINAL DO __INIT__!
        # =================================================================
        self.gravando_video = False
        if self.is_host and "pegador" in self.name.lower():
            self.gravando_video = True
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.video_out = cv2.VideoWriter('partida_vizdoom.avi', fourcc, 35.0, (640, 480))
            print("[VIDEO] Câmera ativada na visão do Pegador!")

    def _read_obs(self):
        state = self.game.get_state()
        # Se vazio, retorna zeros no formato (H, W, 1)
        if state is None:
            return np.zeros(self.observation_space.shape, dtype=np.uint8)

        buf = state.screen_buffer # VizDoom Gray8 devolve (H, W) puro
        
        # 1. Garante o Resize para 160x120 (H=120, W=160)
        target_w = self.dm.screen_w
        target_h = self.dm.screen_h
        
        
        # 2. Garante CHANNEL LAST (H, W, 1)
        # O buffer vem como (120, 160). Adicionamos a dimensão no fim.
        if buf.ndim == 2:
            buf = buf[..., np.newaxis] # Vira (120, 160, 1)
            
        # Caso bizarro de já ter canais (C, H, W) -> converte para (H, W, C)
        elif buf.ndim == 3 and buf.shape[0] <= 3: 
             buf = np.moveaxis(buf, 0, -1) 
            
        return buf

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if not self.game.is_running() or self.game.is_episode_finished():
            try:
                self.game.new_episode()
            except Exception:
                # Fallback simples
                pass
        
        self.shaper.reset(self.game)
        obs = self._read_obs()
        return obs, {"name": self.name}

    def step(self, action: int):
        if self.game.is_player_dead():
            self.game.respawn_player()

        a = self._actions[action].tolist()
        engine_r = float(self.game.make_action(a, self.dm.frame_skip))
        shaped_r = self.shaper.compute(self.game, engine_r)
        done = self.game.is_episode_finished()
        obs = self._read_obs()

        state = self.game.get_state()
        
        # [CÂMERA DE SEGURANÇA]
        if self.gravando_video and state is not None:
            img = state.screen_buffer 
            # O RGB24 já entrega (120, 160, 3), então só invertemos as cores para o OpenCV (BGR)
            frame_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # Dá o zoom e salva
            frame_zoom = cv2.resize(frame_bgr, (640, 480), interpolation=cv2.INTER_NEAREST)
            self.video_out.write(frame_zoom)
        
        return obs, shaped_r, done, False, {"engine_r": engine_r}

    def render(self): pass
    def close(self):
        if self.gravando_video:
            self.video_out.release()
            print("[VIDEO] Arquivo 'partida_vizdoom.avi' salvo e finalizado!")
        try: self.game.close()
        except: pass
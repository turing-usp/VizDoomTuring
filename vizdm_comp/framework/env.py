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

# --- LISTAS DE AÇÕES GLOBAIS ---
actions_deathmatch = np.asarray([
    [0,0,0,0,0,0], [0,0,1,0,0,0], [0,0,0,1,0,0], [0,0,0,0,1,0],
    [0,0,1,0,0,1], [1,0,0,0,0,0], [0,1,0,0,0,0], [0,0,0,0,0,1]
], dtype=np.int32)
buttons_deathmatch = [
    Button.MOVE_LEFT, Button.MOVE_RIGHT, Button.MOVE_FORWARD,
    Button.TURN_LEFT, Button.TURN_RIGHT, Button.ATTACK
]

buttons_tag = [
    Button.MOVE_FORWARD,
    Button.MOVE_BACKWARD,
    Button.TURN_LEFT, 
    Button.TURN_RIGHT,
    Button.SPEED,
    Button.ATTACK 
]
actions_tag = np.asarray([
    [0,0,0,0,0,0], 
    [1,0,0,0,0,0], 
    [1,0,0,0,1,0], 
    [0,1,0,0,0,0], 
    [0,0,1,0,0,0], 
    [0,0,0,1,0,0], 
    [1,0,0,0,0,1], 
    [1,0,0,0,1,1]  
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
        
        # --- MUDANÇA CRUCIAL: NÃO CARREGAMOS MAIS O CONFIG (tag.cfg) ---
        # g.load_config(dm.config_file)  <-- REMOVIDO PARA EVITAR DEADLOCK
        # ---------------------------------------------------------------
        
        # Configuração Manual do WAD (Agora única fonte da verdade)
        if self.dm.wad:
            root_dir = os.getcwd()
            wad_candidate_root = os.path.join(root_dir, self.dm.wad)
            framework_dir = os.path.dirname(os.path.abspath(__file__))
            wad_candidate_fw = os.path.join(framework_dir, os.path.basename(self.dm.wad))

            if os.path.exists(wad_candidate_root):
                # Força ser o IWAD (Jogo Base) para evitar carregar freedoom2.wad junto
                g.set_doom_game_path(wad_candidate_root)
            elif os.path.exists(wad_candidate_fw):
                g.set_doom_game_path(wad_candidate_fw)
            else:
                print(f"[ENV] AVISO: WAD {self.dm.wad} não encontrado, VizDoom tentará padrão.")

        g.set_doom_map(self.dm.map_name)
        g.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
        g.set_screen_format(vzd.ScreenFormat.GRAY8)
        g.set_render_hud(False)
        g.set_render_crosshair(False)
        g.set_window_visible(self.dm.render)

        # Detecta modo de jogo pelo nome do arquivo (string), sem abrir o arquivo
        config_filename = str(dm.config_file).lower()

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

        g.add_available_game_variable(GameVariable.HEALTH)
        g.add_available_game_variable(GameVariable.ARMOR)
        g.add_available_game_variable(GameVariable.AMMO2)
        g.add_available_game_variable(GameVariable.FRAGCOUNT)
        g.add_available_game_variable(GameVariable.DEATHCOUNT)
        g.add_available_game_variable(GameVariable.HITCOUNT)
        g.add_available_game_variable(GameVariable.HITS_TAKEN)

        apply_engine_rewards(g, self.agent.engine_reward)

        g.set_mode(Mode.ASYNC_PLAYER) 

        # Adicionamos -nosound e -noconsole para garantir performance e evitar travar audio
        common = (
            f"-deathmatch +timelimit {self.dm.timelimit_minutes} "
            f"+sv_forcerespawn 1 +sv_noautoaim 1 +sv_respawnprotect 1 +sv_spawnfarthest 1 "
            f"-skill 3 +name {self.name} +colorset {self.agent.colorset} "
            f"-nosound -noconsole "
        )
        if self.is_host:
            args = f"-host {self.dm.total_players} -port {self.dm.port} +viz_connect_timeout 60 "
        else:
            args = f"-join {self.dm.join_ip} -port {self.dm.port} "
        g.add_game_args(args + common)
        
        print(f"[ENV DEBUG] Tentando iniciar VizDoom (g.init) para: {self.name} | Host={self.is_host}")
        try:
            g.init()
            print(f"[ENV DEBUG] VizDoom INICIADO com sucesso para: {self.name}")
        except Exception as e:
            print(f"[ENV DEBUG] ERRO FATAL ao iniciar VizDoom para {self.name}: {e}")
            raise e

        self.shaper = RewardShaper(self.agent.shaping)
        self._last_obs: Optional[np.ndarray] = None

        H, W = self.dm.screen_h, self.dm.screen_w
        self.observation_space = spaces.Box(low=0, high=255, shape=(H, W, 1), dtype=np.uint8)
        self.action_space = spaces.Discrete(len(self._actions))

    def _read_obs(self):
        state = self.game.get_state()
        if state is None:
            return np.zeros(self.observation_space.shape, dtype=np.uint8)

        buf = state.screen_buffer
        target_w = self.dm.screen_w
        target_h = self.dm.screen_h
        
        if buf.shape[0] != target_h or buf.shape[1] != target_w:
             buf = cv2.resize(buf, (target_w, target_h), interpolation=cv2.INTER_AREA)
        
        if buf.ndim == 2:
            buf = buf[..., None]
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
        engine_r = float(self.game.make_action(a, self.dm.frame_skip))
        shaped_r = self.shaper.compute(self.game, engine_r)
        done = self.game.is_episode_finished()
        obs = self._read_obs()
        return obs, shaped_r, done, False, {"engine_r": engine_r}

    def render(self): pass
    def close(self):
        try: self.game.close()
        except: pass
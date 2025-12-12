#!/usr/bin/env python3
from __future__ import annotations

from typing import Optional, Tuple, Dict, Any
import time

import numpy as np
import cv2
import gymnasium as gym
from gymnasium import spaces
import vizdoom as vzd
from vizdoom import Button, GameVariable

from .config import DMConfig, AgentConfig
from .rewards import RewardShaper, apply_engine_rewards


class DoomDMEnv(gym.Env):
    """
    Ambiente VizDoom Deathmatch para RL, em modo multiplayer assíncrono.

    - Jogo roda em modo deathmatch, host/clients configurados via DMConfig.
    - A JANELA de renderização roda em resolução fixa alta (ex.: 640x480).
    - A REDE neural vê frames grayscale redimensionados para a resolução
      especificada em AgentConfig.render_settings.resolution (ex.: RES_120X90).
    - Episódio no sentido do engine é tratado como essencialmente contínuo.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, name: str, is_host: bool, dm: DMConfig, agent: AgentConfig):
        super().__init__()
        self.name = name
        self.is_host = is_host
        self.dm = dm
        self.agent = agent

        # Timelimit: 0 ou negativo -> partida "infinita"
        self._timelimit_minutes: float = float(self.dm.timelimit_minutes)
        self._infinite_match: bool = self._timelimit_minutes <= 0.0

        print(
            f"[ENV] __init__() name={self.name}, is_host={self.is_host}, "
            f"timelimit_minutes={self._timelimit_minutes} "
            f"({'infinite' if self._infinite_match else 'finite'}), "
            f"players={self.dm.total_players}, port={self.dm.port}, "
            f"join_ip={self.dm.join_ip}"
        )

        # ------------------------------------------------------------------
        # Configuração do DoomGame
        # ------------------------------------------------------------------
        g = self.game = vzd.DoomGame()
        g.load_config("cig.cfg")
        g.set_doom_map(self.dm.map_name)

        # ------------------------------------------------------------------
        # Config visual: janela em alta resolução fixa + resolução da REDE
        # ------------------------------------------------------------------
        rs = self.agent.render_settings

        # 1) Resolução da JANELA (tela que você vê) – fixa em alta resolução
        try:
            window_res_enum = vzd.ScreenResolution.RES_640X480
        except AttributeError:
            # Fallback em builds que não tenham 640x480
            print(
                "[ENV][WARN] RES_640X480 não disponível, usando RES_160X120 para janela."
            )
            window_res_enum = vzd.ScreenResolution.RES_160X120
        g.set_screen_resolution(window_res_enum)

        # 2) Formato da janela (RGB/GRAY) conforme YAML
        try:
            fmt_enum = getattr(vzd.ScreenFormat, rs.format)
        except AttributeError:
            print(
                f"[ENV][WARN] Formato inválido em render_settings: {rs.format!r}, "
                "usando GRAY8."
            )
            fmt_enum = vzd.ScreenFormat.GRAY8
        g.set_screen_format(fmt_enum)

        # 3) Resolução usada pela REDE (NET_W/NET_H) a partir de rs.resolution
        #    Espera string tipo "RES_120X90" no YAML.
        default_net_w, default_net_h = 160, 120
        net_w, net_h = default_net_w, default_net_h
        try:
            if isinstance(rs.resolution, str) and rs.resolution.startswith("RES_"):
                dims = rs.resolution[4:].split("X")
                if len(dims) == 2:
                    net_w = int(dims[0])
                    net_h = int(dims[1])
                else:
                    print(
                        f"[ENV][WARN] render_settings.resolution={rs.resolution!r} "
                        f"não está no formato RES_WXH. Usando {default_net_w}x{default_net_h}."
                    )
            else:
                print(
                    f"[ENV][WARN] render_settings.resolution={rs.resolution!r} "
                    f"não começa com 'RES_'. Usando {default_net_w}x{default_net_h}."
                )
        except Exception as e:
            print(
                f"[ENV][WARN] Falha ao interpretar render_settings.resolution={rs.resolution!r}: {e!r}. "
                f"Usando {default_net_w}x{default_net_h} para a rede."
            )
            net_w, net_h = default_net_w, default_net_h

        # Finalmente, define a resolução que a REDE vê
        self.NET_W = net_w
        self.NET_H = net_h
        self.NET_C = 1  # grayscale
        print(
            f"[ENV] Resolução da REDE (obs): {self.NET_W}x{self.NET_H}x{self.NET_C} "
            f"(janela renderiza em {window_res_enum})"
        )

        g.set_render_hud(bool(rs.hud))
        g.set_render_crosshair(False)
        g.set_render_weapon(True)
        g.set_window_visible(self.dm.render)

        # Botões disponíveis
        g.set_available_buttons(
            [
                Button.MOVE_LEFT,
                Button.MOVE_RIGHT,
                Button.MOVE_FORWARD,
                Button.TURN_LEFT,
                Button.TURN_RIGHT,
                Button.ATTACK,
            ]
        )

        # Variáveis para shaping/infos
        g.add_available_game_variable(GameVariable.HEALTH)
        g.add_available_game_variable(GameVariable.ARMOR)
        g.add_available_game_variable(GameVariable.AMMO2)
        g.add_available_game_variable(GameVariable.FRAGCOUNT)
        g.add_available_game_variable(GameVariable.DEATHCOUNT)
        g.add_available_game_variable(GameVariable.HITCOUNT)
        g.add_available_game_variable(GameVariable.HITS_TAKEN)

        # Usa bloco unificado de recompensa (engine)
        apply_engine_rewards(g, self.agent.reward.engine)

        # Deathmatch assíncrono e acelerado
        # Treino = ticrate alto (rápido), Play = ticrate normal de Doom (~35 tics/s).
        if self.agent.train:
            g.set_ticrate(50)
        else:
            g.set_ticrate(35)

        g.set_episode_timeout(0)
        g.set_episode_start_time(0)

        # ------------------------------------------------------------------
        # Args de deathmatch multiplayer
        # ------------------------------------------------------------------
        if self._infinite_match:
            timelimit_arg = "+timelimit 0 "
        else:
            timelimit_arg = f"+timelimit {self._timelimit_minutes} "

        common = (
            f"-deathmatch {timelimit_arg}"
            f"+fraglimit 0 +sv_forcerespawn 1 +sv_noautoaim 1 "
            f"+sv_respawnprotect 1 +sv_spawnfarthest 1 "
            f"-skill 3 +name {self.name} +colorset {self.agent.colorset} "
        )

        if self.is_host:
            args = (
                f"-host {self.dm.total_players} "
                f"-port {self.dm.port} "
                f"+viz_connect_timeout 300 "
                f"+sv_noautoaim 1 "
            )
        else:
            args = (
                f"-join {self.dm.join_ip} "
                f"-port {self.dm.port} "
                f"+viz_connect_timeout 300 "
                f"+cl_connect_timeout 300 "
            )

        g.add_game_args(args + common)

        print(f"[ENV] Chamando game.init() (is_host={self.is_host}, name={self.name})")
        g.init()
        print(f"[ENV] game.init() concluído (is_host={self.is_host}, name={self.name})")

        # ------------------------------------------------------------------
        # Ações discretas
        # ------------------------------------------------------------------
        self._actions = np.asarray(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 1],
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1],
            ],
            dtype=np.int32,
        )

        # Space que a REDE vê: NET_H x NET_W x 1, uint8
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.NET_H, self.NET_W, self.NET_C),
            dtype=np.uint8,
        )
        self.action_space = spaces.Discrete(len(self._actions))

        # Shaper usa reward.shaping
        self.shaper = RewardShaper(self.agent.reward.shaping)
        self._last_obs: Optional[np.ndarray] = None

        # Contador de "episódios" do engine (para logging)
        self._engine_episode_count: int = 0

    # ----------------------------------------------------------------------
    # Leitura e normalização de observação
    # ----------------------------------------------------------------------
    def _read_obs(self) -> np.ndarray:
        """
        Lê frame do VizDoom, converte para grayscale NET_H x NET_W x 1 (uint8)
        e mantém um último frame em cache para casos em que o estado ainda
        não está disponível.
        """
        st = self.game.get_state()

        # Sem estado disponível -> reaproveita último frame ou retorna preto
        if st is None or st.screen_buffer is None:
            if self._last_obs is not None:
                return self._last_obs
            obs = np.zeros((self.NET_H, self.NET_W, self.NET_C), dtype=np.uint8)
            self._last_obs = obs
            return obs

        img = st.screen_buffer  # pode vir como (C,H,W) ou (H,W) ou (H,W,C)

        # Garante layout HWC
        if img.ndim == 3:
            # Muitas builds de VizDoom retornam (C,H,W)
            if img.shape[0] in (1, 3):
                img = img.transpose(1, 2, 0)

            # Se for RGB, converte para grayscale
            if img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Agora img é (H,W) ou (H,W,1) (idealmente)
        if img.ndim == 2:
            # (H,W) grayscale
            if img.shape[0] != self.NET_H or img.shape[1] != self.NET_W:
                img = cv2.resize(
                    img, (self.NET_W, self.NET_H), interpolation=cv2.INTER_AREA
                )
            img = img[..., np.newaxis]
        elif img.ndim == 3:
            # (H,W,C)
            if img.shape[0] != self.NET_H or img.shape[1] != self.NET_W:
                img = cv2.resize(
                    img, (self.NET_W, self.NET_H), interpolation=cv2.INTER_AREA
                )

            if img.shape[2] != 1:
                # Garantir canal único
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                img = img[..., np.newaxis]
        else:
            # Formato inesperado -> fallback preto
            img = np.zeros((self.NET_H, self.NET_W, self.NET_C), dtype=np.uint8)

        img = np.ascontiguousarray(img, dtype=np.uint8)
        self._last_obs = img
        return img

    # ----------------------------------------------------------------------
    # API Gymnasium
    # ----------------------------------------------------------------------
    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset sem chamar new_episode() na primeira vez para evitar travamento.
        """
        print(
            f"[ENV] reset() chamado (is_host={self.is_host}, name={self.name}, "
            f"engine_episode_count={self._engine_episode_count})"
        )
        super().reset(seed=seed)
        try:
            g = self.game

            if not g.is_running():
                raise RuntimeError(
                    f"[ENV] reset(): game não está rodando "
                    f"(is_host={self.is_host}, name={self.name})"
                )

            if self._engine_episode_count == 0:
                # Primeiro reset: usa episódio criado por game.init()
                self._engine_episode_count = 1
                print(
                    f"[ENV] reset(): usando episódio criado por init() "
                    f"(engine_episode_count={self._engine_episode_count}, "
                    f"is_host={self.is_host}, name={self.name})"
                )
            else:
                # Resets subsequentes: não chamamos new_episode() para evitar travas.
                print(
                    f"[ENV] reset(): reusando episódio existente "
                    f"(engine_episode_count={self._engine_episode_count}, "
                    f"is_host={self.is_host}, name={self.name})"
                )

            # Re-sincroniza o RewardShaper com o estado atual
            self.shaper.reset(g)
            obs = self._read_obs()

            print(
                f"[ENV] reset() concluído (is_host={self.is_host}, name={self.name}, "
                f"engine_episode_count={self._engine_episode_count}, "
                f"obs_shape={getattr(obs, 'shape', None)})"
            )
            return obs, {"name": self.name}

        except Exception as e:
            print(
                f"[ENV][ERROR] Exceção em reset() "
                f"(is_host={self.is_host}, name={self.name}): {e!r}"
            )
            raise

    def step(self, action: int):
        """
        Passo do ambiente.
        """
        try:
            if self.game.is_player_dead():
                self.game.respawn_player()

            a = self._actions[action].tolist()
            engine_r = float(self.game.make_action(a, self.dm.frame_skip))
            shaped_r = self.shaper.compute(self.game, engine_r)

            if self.game.is_episode_finished():
                print(
                    f"[ENV][DEBUG] is_episode_finished()==True em step() "
                    f"(engine_episode_count={self._engine_episode_count}, "
                    f"is_host={self.is_host}, name={self.name})"
                )
                t0 = time.monotonic()
                self.game.new_episode()
                self._engine_episode_count += 1
                self.shaper.reset(self.game)
                elapsed = time.monotonic() - t0
                print(
                    f"[ENV][DEBUG] new_episode() em step() OK em {elapsed:.3f}s "
                    f"(engine_episode_count={self._engine_episode_count}, "
                    f"is_host={self.is_host}, name={self.name})"
                )

            obs = self._read_obs()

            info: Dict[str, Any] = {
                "engine_r": engine_r,
                "engine_episode_count": self._engine_episode_count,
            }

            # Tarefa contínua: não sinalizamos término para o SB3
            terminated = False
            truncated = False

            return obs, shaped_r, terminated, truncated, info

        except Exception as e:
            print(
                f"[ENV][ERROR] Exceção em step(action={action}) "
                f"(is_host={self.is_host}, name={self.name}): {e!r}"
            )
            raise

    def render(self):
        # Render já é controlado por VizDoom (janela visível ou não via dm.render).
        pass

    def close(self):
        print(f"[ENV] close() chamado (is_host={self.is_host}, name={self.name})")
        try:
            self.game.close()
        except Exception as e:
            print(
                f"[ENV][WARN] Exceção em game.close() "
                f"(is_host={self.is_host}, name={self.name}): {e!r}"
            )

#!/usr/bin/env python3
from __future__ import annotations

import os
import time
from typing import Optional, Tuple, Dict, Any, List

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

    Sem lógica de arma:
    - Não força arma preferida
    - Não inclui botões SELECT_WEAPON*
    - A arma usada é a do mapa/mod + autoswitch/pickups do Doom.

    Otimização headless:
    - Se dm.render == False: seta screen_resolution = resolução da REDE (rs.resolution)
      e screen_format = GRAY8, e desliga HUD/weapon/crosshair para evitar custo e resize.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, name: str, is_host: bool, dm: DMConfig, agent: AgentConfig):
        super().__init__()
        self.name = name
        self.is_host = is_host
        self.dm = dm
        self.agent = agent

        # Evita OpenCV criar threads em excesso
        try:
            cv2.setNumThreads(0)
        except Exception:
            pass

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
        # DoomGame
        # ------------------------------------------------------------------
        g = self.game = vzd.DoomGame()

        # cig.cfg precisa estar acessível (CWD) ou use caminho absoluto
        g.load_config("cig.cfg")
        g.set_doom_map(self.dm.map_name)

        # ------------------------------------------------------------------
        # Render settings (rede vs janela)
        # ------------------------------------------------------------------
        rs = self.agent.render_settings

        # Parse "RES_WXH" do YAML
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

        self.NET_W = int(net_w)
        self.NET_H = int(net_h)
        self.NET_C = 1  # rede sempre grayscale

        def _get_res_enum(res_name: str, fallback: str = "RES_160X120") -> "vzd.ScreenResolution":
            if hasattr(vzd.ScreenResolution, res_name):
                return getattr(vzd.ScreenResolution, res_name)
            return getattr(vzd.ScreenResolution, fallback, vzd.ScreenResolution.RES_160X120)

        # Resolução enum da REDE (ex: "RES_120X90")
        net_res_name = rs.resolution if isinstance(rs.resolution, str) else "RES_160X120"
        net_res_enum = _get_res_enum(net_res_name)

        if bool(self.dm.render):
            # Janela (debug): alta resolução
            window_res_enum = _get_res_enum("RES_640X480", fallback="RES_160X120")

            # Formato conforme YAML (debug)
            try:
                fmt_enum = getattr(vzd.ScreenFormat, rs.format)
            except Exception:
                print(f"[ENV][WARN] Formato inválido em render_settings: {rs.format!r}, usando GRAY8.")
                fmt_enum = vzd.ScreenFormat.GRAY8

            g.set_screen_resolution(window_res_enum)
            g.set_screen_format(fmt_enum)

            g.set_render_hud(bool(rs.hud))
            g.set_render_crosshair(False)
            g.set_render_weapon(True)
            g.set_window_visible(True)

        else:
            # Headless (treino): igual rede, GRAY8, sem overlays -> evita cv2.cvtColor/resize
            g.set_screen_resolution(net_res_enum)
            g.set_screen_format(vzd.ScreenFormat.GRAY8)

            g.set_render_hud(False)
            g.set_render_crosshair(False)
            g.set_render_weapon(False)
            g.set_window_visible(False)

            # Como headless é GRAY8, não precisamos de cv2 para converter
            # (mas mantemos fallback seguro caso algo venha diferente)

        print(
            f"[ENV] Resolução da REDE (obs): {self.NET_W}x{self.NET_H}x{self.NET_C} "
            f"(screen_buffer em {('window' if self.dm.render else 'net')}={net_res_enum}, render={self.dm.render})"
        )

        # ------------------------------------------------------------------
        # Botões disponíveis (sem SELECT_WEAPON*)
        # ------------------------------------------------------------------
        self._buttons: List[Button] = [
            Button.MOVE_LEFT,
            Button.MOVE_RIGHT,
            Button.MOVE_FORWARD,
            Button.TURN_LEFT,
            Button.TURN_RIGHT,
            Button.ATTACK,
        ]
        g.set_available_buttons(self._buttons)

        # ------------------------------------------------------------------
        # Variáveis para shaping/infos
        # ------------------------------------------------------------------
        g.add_available_game_variable(GameVariable.HEALTH)
        g.add_available_game_variable(GameVariable.ARMOR)
        g.add_available_game_variable(GameVariable.AMMO2)
        g.add_available_game_variable(GameVariable.FRAGCOUNT)
        g.add_available_game_variable(GameVariable.DEATHCOUNT)
        g.add_available_game_variable(GameVariable.HITCOUNT)
        g.add_available_game_variable(GameVariable.HITS_TAKEN)

        apply_engine_rewards(g, self.agent.reward.engine)

        # Ticrate
        g.set_ticrate(30 if self.agent.train else 35)
        g.set_episode_timeout(0)
        g.set_episode_start_time(0)

        # ------------------------------------------------------------------
        # Args multiplayer (+ WAD extra opcional)
        # ------------------------------------------------------------------
        timelimit_arg = "+timelimit 0 " if self._infinite_match else f"+timelimit {self._timelimit_minutes} "

        # WAD/PK3 extra (não substitui o doom_scenario_path do cfg; só adiciona conteúdo)
        extra_file_arg = ""
        wad = getattr(self.dm, "wad", None)
        if wad:
            wad = str(wad).strip()
            if wad:
                wad_path = wad
                if not os.path.isfile(wad_path):
                    # tenta achar em framework/maps (relativo ao arquivo env.py)
                    base_dir = os.path.dirname(__file__)
                    maps_dir = os.path.join(base_dir, "maps")
                    cand = os.path.join(maps_dir, wad)
                    if os.path.isfile(cand):
                        wad_path = cand
                    else:
                        # tenta com extensões
                        for ext in (".wad", ".pk3"):
                            cand2 = cand if cand.lower().endswith(ext) else cand + ext
                            if os.path.isfile(cand2):
                                wad_path = cand2
                                break

                if os.path.isfile(wad_path):
                    extra_file_arg = f"-file {wad_path} "
                    print(f"[ENV] Extra WAD/PK3: {wad_path}")
                else:
                    print(f"[ENV][WARN] DMConfig.wad={wad!r} não encontrado. Ignorando.")

        common = (
            f"{extra_file_arg}"
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
        # Ações discretas (8 ações) -> 6 botões
        # ------------------------------------------------------------------
        base_actions = np.asarray(
            [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],  # forward
                [0, 0, 0, 1, 0, 0],  # turn_left
                [0, 0, 0, 0, 1, 0],  # turn_right
                [0, 0, 0, 0, 0, 1],  # forward+attack
                [1, 0, 0, 0, 0, 0],  # move_left
                [0, 1, 0, 0, 0, 0],  # move_right
                [0, 0, 0, 0, 0, 1],  # attack
            ],
            dtype=np.int32,
        )

        n_btn = len(self._buttons)
        if base_actions.shape[1] != n_btn:
            raise RuntimeError(
                f"[ENV] Inconsistência: base_actions tem {base_actions.shape[1]} colunas, "
                f"mas available_buttons tem {n_btn}."
            )

        self._actions = base_actions

        # Spaces
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.NET_H, self.NET_W, self.NET_C),
            dtype=np.uint8,
        )
        self.action_space = spaces.Discrete(len(self._actions))

        # Shaper
        self.shaper = RewardShaper(self.agent.reward.shaping)
        self._last_obs: Optional[np.ndarray] = None
        self._engine_episode_count: int = 0

    # ----------------------------------------------------------------------
    # Obs: SEMPRE (NET_H, NET_W, 1) uint8
    # ----------------------------------------------------------------------
    def _read_obs(self) -> np.ndarray:
        st = self.game.get_state()

        if st is None or st.screen_buffer is None:
            if self._last_obs is not None:
                return self._last_obs
            obs = np.zeros((self.NET_H, self.NET_W, 1), dtype=np.uint8)
            self._last_obs = obs
            return obs

        img = st.screen_buffer  # pode vir CHW/HWC/HW

        # Caso ideal (headless): GRAY8 + screen_resolution == NET => img 2D (H,W)
        if img.ndim == 2:
            gray = img

        elif img.ndim == 3:
            # CHW -> HWC (quando vem CHW)
            if img.shape[0] in (1, 3) and (img.shape[1] != 1 and img.shape[2] != 1):
                img = img.transpose(1, 2, 0)

            if img.shape[2] == 1:
                gray = img[:, :, 0]
            elif img.shape[2] == 3:
                # fallback (render True com RGB)
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                gray = img[:, :, 0]

        else:
            gray = np.zeros((self.NET_H, self.NET_W), dtype=np.uint8)

        # Fallback: só resize se necessário (idealmente nunca acontece em render=False)
        if gray.shape[0] != self.NET_H or gray.shape[1] != self.NET_W:
            gray = cv2.resize(gray, (self.NET_W, self.NET_H), interpolation=cv2.INTER_AREA)

        obs = np.ascontiguousarray(gray, dtype=np.uint8)[..., None]  # (H,W,1)

        if obs.shape != (self.NET_H, self.NET_W, 1):
            raise RuntimeError(f"[ENV] Obs inválida {obs.shape}, esperado {(self.NET_H, self.NET_W, 1)}")

        self._last_obs = obs
        return obs

    # ----------------------------------------------------------------------
    # API Gymnasium
    # ----------------------------------------------------------------------
    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        print(
            f"[ENV] reset() chamado (is_host={self.is_host}, name={self.name}, "
            f"engine_episode_count={self._engine_episode_count})"
        )
        super().reset(seed=seed)

        g = self.game
        if not g.is_running():
            raise RuntimeError(f"[ENV] reset(): game não está rodando (is_host={self.is_host}, name={self.name})")

        if self._engine_episode_count == 0:
            self._engine_episode_count = 1
            print(
                f"[ENV] reset(): usando episódio criado por init() "
                f"(engine_episode_count={self._engine_episode_count}, is_host={self.is_host}, name={self.name})"
            )
        else:
            print(
                f"[ENV] reset(): reusando episódio existente "
                f"(engine_episode_count={self._engine_episode_count}, is_host={self.is_host}, name={self.name})"
            )

        self.shaper.reset(g)
        obs = self._read_obs()

        print(
            f"[ENV] reset() concluído (is_host={self.is_host}, name={self.name}, "
            f"engine_episode_count={self._engine_episode_count}, obs_shape={obs.shape})"
        )
        return obs, {"name": self.name}

    def step(self, action: int):
        try:
            if self.game.is_player_dead():
                self.game.respawn_player()

            a = self._actions[int(action)].tolist()
            engine_r = float(self.game.make_action(a, int(self.dm.frame_skip)))
            shaped_r = self.shaper.compute(self.game, engine_r)

            if self.game.is_episode_finished():
                print(
                    f"[ENV][DEBUG] is_episode_finished()==True em step() "
                    f"(engine_episode_count={self._engine_episode_count}, is_host={self.is_host}, name={self.name})"
                )
                t0 = time.monotonic()
                self.game.new_episode()
                self._engine_episode_count += 1
                self.shaper.reset(self.game)
                elapsed = time.monotonic() - t0
                print(
                    f"[ENV][DEBUG] new_episode() OK em {elapsed:.3f}s "
                    f"(engine_episode_count={self._engine_episode_count}, is_host={self.is_host}, name={self.name})"
                )

            obs = self._read_obs()

            info: Dict[str, Any] = {
                "engine_r": engine_r,
                "engine_episode_count": self._engine_episode_count,
            }

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

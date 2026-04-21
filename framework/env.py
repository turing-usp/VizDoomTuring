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
from .rewards import (
    RewardShaper,
    ContextualRewardShaper,
    MotionSnapshot,
    StepVarsSnapshot,
    apply_engine_rewards,
    capture_vars,
)


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

    @staticmethod
    def _resolve_map_asset(path_or_name: Optional[str]) -> Optional[str]:
        if not path_or_name:
            return None

        raw = str(path_or_name).strip()
        if not raw:
            return None

        if os.path.isfile(raw):
            return raw

        maps_dir = os.path.join(os.path.dirname(__file__), "maps")
        direct = os.path.join(maps_dir, raw)
        if os.path.isfile(direct):
            return direct

        base, ext = os.path.splitext(raw)
        if not ext:
            for candidate_ext in (".cfg", ".wad", ".pk3"):
                candidate = os.path.join(maps_dir, raw + candidate_ext)
                if os.path.isfile(candidate):
                    return candidate

        return raw

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
        self._render_agent_view: bool = bool(getattr(self.dm, "render_agent_view", False))
        self._agent_view_window_name: str = f"agent_view::{self.name}"
        self._labels_enabled: bool = False

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
        scenario_path = self._resolve_map_asset(getattr(self.dm, "scenario", None))
        if scenario_path and str(scenario_path).lower().endswith(".cfg"):
            print(f"[ENV] Scenario CFG: {scenario_path}")
            g.load_config(str(scenario_path))
        else:
            print("[ENV] Scenario CFG: cig.cfg")
            g.load_config("cig.cfg")

        if scenario_path and str(scenario_path).lower().endswith((".wad", ".pk3")):
            print(f"[ENV] Scenario base WAD/PK3: {scenario_path}")
            g.set_doom_scenario_path(str(scenario_path))
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
            if self._render_agent_view:
                # Host_agent: a janela do VizDoom fica oculta e mostramos a observação processada via OpenCV
                g.set_screen_resolution(net_res_enum)
                g.set_screen_format(vzd.ScreenFormat.GRAY8)
                g.set_render_hud(bool(rs.hud))
                g.set_render_crosshair(bool(getattr(rs, "crosshair", False)))
                g.set_render_weapon(bool(getattr(rs, "weapon", False)))
                g.set_window_visible(False)
            else:
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
                g.set_render_crosshair(bool(getattr(rs, "crosshair", False)))
                g.set_render_weapon(bool(getattr(rs, "weapon", True)))
                g.set_window_visible(True)

        else:
            # Headless (treino): igual rede, GRAY8, overlays conforme YAML.
            g.set_screen_resolution(net_res_enum)
            g.set_screen_format(vzd.ScreenFormat.GRAY8)

            g.set_render_hud(bool(rs.hud))
            g.set_render_crosshair(bool(getattr(rs, "crosshair", False)))
            g.set_render_weapon(bool(getattr(rs, "weapon", False)))
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
            Button.MOVE_BACKWARD,
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
        g.add_available_game_variable(GameVariable.DAMAGECOUNT)
        g.add_available_game_variable(GameVariable.FRAGCOUNT)
        g.add_available_game_variable(GameVariable.DEATHCOUNT)
        g.add_available_game_variable(GameVariable.HITCOUNT)
        g.add_available_game_variable(GameVariable.HITS_TAKEN)
        g.add_available_game_variable(GameVariable.POSITION_X)
        g.add_available_game_variable(GameVariable.POSITION_Y)
        g.add_available_game_variable(GameVariable.ANGLE)
        g.add_available_game_variable(GameVariable.SELECTED_WEAPON)
        g.add_available_game_variable(GameVariable.SELECTED_WEAPON_AMMO)
        g.add_available_game_variable(GameVariable.VELOCITY_X)
        g.add_available_game_variable(GameVariable.VELOCITY_Y)
        g.add_available_game_variable(GameVariable.VELOCITY_Z)

        apply_engine_rewards(g, self.agent.reward.engine)

        if hasattr(g, "set_labels_buffer_enabled"):
            g.set_labels_buffer_enabled(True)
            self._labels_enabled = True
        else:
            print("[ENV][WARN] Labels buffer nao suportado nesta versao do ViZDoom. enemy_in_view sera ignorado.")

        # Ticrate
        ticrate = int(getattr(self.dm, "ticrate", 30 if self.agent.train else 35))
        g.set_ticrate(ticrate)
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
            wad_path = self._resolve_map_asset(wad)
            if wad_path:
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
        print(
            f"[ENV] Runtime: scenario={scenario_path!r}, map={self.dm.map_name!r}, "
            f"frame_skip={int(self.dm.frame_skip)}, ticrate={ticrate}, render={bool(self.dm.render)}, "
            f"render_agent_view={self._render_agent_view}",
            flush=True,
        )

        print(f"[ENV] Chamando game.init() (is_host={self.is_host}, name={self.name})")
        g.init()
        print(f"[ENV] game.init() concluído (is_host={self.is_host}, name={self.name})")

        # ------------------------------------------------------------------
        # Ações discretas (8 ações) -> 6 botões
        # ------------------------------------------------------------------
        base_actions = np.asarray(
            [
                [0, 0, 0, 0, 0, 0, 0],  # noop
                [0, 0, 1, 0, 0, 0, 0],  # forward
                [0, 0, 0, 1, 0, 0, 0],  # backward
                [0, 0, 0, 0, 1, 0, 0],  # turn_left
                [0, 0, 0, 0, 0, 1, 0],  # turn_right
                [1, 0, 0, 0, 0, 0, 0],  # move_left
                [0, 1, 0, 0, 0, 0, 0],  # move_right
                [0, 0, 0, 0, 0, 0, 1],  # attack
                [0, 0, 1, 0, 0, 0, 1],  # forward+attack
                [0, 0, 0, 1, 0, 0, 1],  # backward+attack
                [0, 0, 0, 0, 1, 0, 1],  # turn_left+attack
                [0, 0, 0, 0, 0, 1, 1],  # turn_right+attack
                [1, 0, 0, 0, 0, 0, 1],  # move_left+attack
                [0, 1, 0, 0, 0, 0, 1],  # move_right+attack
                [0, 0, 1, 0, 1, 0, 0],  # forward+turn_left
                [0, 0, 1, 0, 0, 1, 0],  # forward+turn_right
                [0, 0, 1, 0, 1, 0, 1],  # forward+turn_left+attack
                [0, 0, 1, 0, 0, 1, 1],  # forward+turn_right+attack
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
        self.STATE_DIM = 18
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(
                    low=0,
                    high=255,
                    shape=(self.NET_C, self.NET_H, self.NET_W),
                    dtype=np.uint8,
                ),
                "state": spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(self.STATE_DIM,),
                    dtype=np.float32,
                ),
            }
        )
        self.action_space = spaces.Discrete(len(self._actions))

        # Shaper
        self.shaper = RewardShaper(self.agent.reward.shaping)
        self.contextual_shaper = ContextualRewardShaper(
            self.agent.reward.shaping,
            self.agent.reward.wall_stuck,
            self.agent.reward.enemy_in_view,
        )
        self._last_obs: Optional[np.ndarray] = None
        self._last_state_vars: Optional[Dict[str, float]] = None
        self._engine_episode_count: int = 0

    def _motion_snapshot(self) -> MotionSnapshot:
        return MotionSnapshot(
            x=float(self.game.get_game_variable(GameVariable.POSITION_X)),
            y=float(self.game.get_game_variable(GameVariable.POSITION_Y)),
            angle_deg=float(self.game.get_game_variable(GameVariable.ANGLE)),
        )

    def _show_agent_view_window(self, obs: np.ndarray) -> None:
        if not self._render_agent_view:
            return
        try:
            frame = np.asarray(obs)
            if frame.ndim == 3 and frame.shape[0] == 1:
                frame = frame[0]
            elif frame.ndim == 3 and frame.shape[2] == 1:
                frame = frame[:, :, 0]
            frame = np.ascontiguousarray(frame, dtype=np.uint8)
            cv2.imshow(self._agent_view_window_name, frame)
            cv2.waitKey(1)
        except Exception as e:
            print(f"[ENV][WARN] Falha ao desenhar host_agent view: {e!r}")

    # ----------------------------------------------------------------------
    # Obs: SEMPRE (1, NET_H, NET_W) uint8
    # ----------------------------------------------------------------------
    def _read_obs(self, state: Optional[Any] = None) -> np.ndarray:
        st = state if state is not None else self.game.get_state()

        if st is None or st.screen_buffer is None:
            if self._last_obs is not None:
                return self._last_obs
            obs = np.zeros((1, self.NET_H, self.NET_W), dtype=np.uint8)
            self._last_obs = obs
            return obs

        img = st.screen_buffer

        if img.ndim == 2:
            gray = img
        elif img.ndim == 3:
            if img.shape[0] in (1, 3) and (img.shape[1] != 1 and img.shape[2] != 1):
                img = img.transpose(1, 2, 0)

            if img.shape[2] == 1:
                gray = img[:, :, 0]
            elif img.shape[2] == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                gray = img[:, :, 0]
        else:
            gray = np.zeros((self.NET_H, self.NET_W), dtype=np.uint8)

        source_h, source_w = int(gray.shape[0]), int(gray.shape[1])
        if source_h != self.NET_H or source_w != self.NET_W:
            gray = cv2.resize(gray, (self.NET_W, self.NET_H), interpolation=cv2.INTER_AREA)

        gray = self._draw_enemy_outline(gray, st, source_w=source_w, source_h=source_h)

        obs = np.ascontiguousarray(gray[None, ...], dtype=np.uint8)

        if obs.shape != (1, self.NET_H, self.NET_W):
            raise RuntimeError(f"[ENV] Obs invalida {obs.shape}, esperado {(1, self.NET_H, self.NET_W)}")

        self._last_obs = obs
        return obs

    def _draw_enemy_outline(
        self,
        gray: np.ndarray,
        state: Optional[Any],
        *,
        source_w: int,
        source_h: int,
    ) -> np.ndarray:
        if not bool(getattr(self.agent.render_settings, "enemy_outline", False)):
            return gray
        if state is None or not self._labels_enabled:
            return gray

        labels = getattr(state, "labels", None) or []
        if not labels:
            return gray

        frame = np.ascontiguousarray(gray, dtype=np.uint8).copy()
        value = int(max(0, min(255, int(getattr(self.agent.render_settings, "enemy_outline_value", 255)))))
        thickness = max(1, int(getattr(self.agent.render_settings, "enemy_outline_thickness", 2)))
        pattern = str(getattr(self.agent.render_settings, "enemy_outline_pattern", "solid")).strip().lower()
        scale_x = float(self.NET_W) / max(1.0, float(source_w))
        scale_y = float(self.NET_H) / max(1.0, float(source_h))

        for label in labels:
            category = str(getattr(label, "object_category", "") or "")
            if category != "Player":
                continue

            x = int(round(float(getattr(label, "x", 0.0) or 0.0) * scale_x))
            y = int(round(float(getattr(label, "y", 0.0) or 0.0) * scale_y))
            w = int(round(float(getattr(label, "width", 0.0) or 0.0) * scale_x))
            h = int(round(float(getattr(label, "height", 0.0) or 0.0) * scale_y))
            if w <= 1 or h <= 1:
                continue

            x1 = max(0, min(self.NET_W - 1, x))
            y1 = max(0, min(self.NET_H - 1, y))
            x2 = max(0, min(self.NET_W - 1, x + w))
            y2 = max(0, min(self.NET_H - 1, y + h))
            if x2 <= x1 or y2 <= y1:
                continue

            if pattern == "checker":
                self._draw_checker_rect(frame, x1, y1, x2, y2, thickness=thickness)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color=value, thickness=thickness)

        return frame

    @staticmethod
    def _draw_checker_rect(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, *, thickness: int) -> None:
        h, w = frame.shape[:2]
        thickness = max(1, int(thickness))

        for offset in range(thickness):
            left = max(0, min(w - 1, x1 + offset))
            right = max(0, min(w - 1, x2 - offset))
            top = max(0, min(h - 1, y1 + offset))
            bottom = max(0, min(h - 1, y2 - offset))
            if right < left or bottom < top:
                break

            xs = np.arange(left, right + 1)
            ys = np.arange(top, bottom + 1)

            top_values = np.where(((xs + top) & 1) == 0, 255, 0).astype(np.uint8)
            bottom_values = np.where(((xs + bottom) & 1) == 0, 255, 0).astype(np.uint8)
            left_values = np.where(((left + ys) & 1) == 0, 255, 0).astype(np.uint8)
            right_values = np.where(((right + ys) & 1) == 0, 255, 0).astype(np.uint8)

            frame[top, left : right + 1] = top_values
            frame[bottom, left : right + 1] = bottom_values
            frame[top : bottom + 1, left] = left_values
            frame[top : bottom + 1, right] = right_values

    @staticmethod
    def _clip01(value: float) -> float:
        return float(max(0.0, min(1.0, value)))

    @staticmethod
    def _clip11(value: float) -> float:
        return float(max(-1.0, min(1.0, value)))

    def _state_vars_snapshot(self) -> Dict[str, float]:
        g = self.game
        vx = float(g.get_game_variable(GameVariable.VELOCITY_X))
        vy = float(g.get_game_variable(GameVariable.VELOCITY_Y))
        return {
            "health": float(g.get_game_variable(GameVariable.HEALTH)),
            "armor": float(g.get_game_variable(GameVariable.ARMOR)),
            "ammo2": float(g.get_game_variable(GameVariable.AMMO2)),
            "selected_weapon": float(g.get_game_variable(GameVariable.SELECTED_WEAPON)),
            "selected_weapon_ammo": float(g.get_game_variable(GameVariable.SELECTED_WEAPON_AMMO)),
            "damage": float(g.get_game_variable(GameVariable.DAMAGECOUNT)),
            "frags": float(g.get_game_variable(GameVariable.FRAGCOUNT)),
            "deaths": float(g.get_game_variable(GameVariable.DEATHCOUNT)),
            "hits": float(g.get_game_variable(GameVariable.HITCOUNT)),
            "hits_taken": float(g.get_game_variable(GameVariable.HITS_TAKEN)),
            "vx": vx,
            "vy": vy,
            "vz": float(g.get_game_variable(GameVariable.VELOCITY_Z)),
            "speed_xy": float((vx * vx + vy * vy) ** 0.5),
        }

    def _enemy_state_features(self, state: Optional[Any]) -> Tuple[float, float, float, float, float]:
        if state is None or not self._labels_enabled:
            return 0.0, 0.0, 0.5, 0.5, 0.0

        labels = getattr(state, "labels", None) or []
        enemies: List[Tuple[float, float, float, float, float]] = []
        screen_area = max(1.0, float(self.NET_W * self.NET_H))
        for label in labels:
            category = str(getattr(label, "object_category", "") or "")
            if category != "Player":
                continue
            x = float(getattr(label, "x", 0.0) or 0.0)
            y = float(getattr(label, "y", 0.0) or 0.0)
            w = max(0.0, float(getattr(label, "width", 0.0) or 0.0))
            h = max(0.0, float(getattr(label, "height", 0.0) or 0.0))
            area_ratio = self._clip01((w * h) / screen_area)
            cx = self._clip01((x + w * 0.5) / max(1.0, float(self.NET_W)))
            cy = self._clip01((y + h * 0.5) / max(1.0, float(self.NET_H)))
            enemies.append((area_ratio, cx, cy, w, h))

        if not enemies:
            return 0.0, 0.0, 0.5, 0.5, 0.0

        largest = max(enemies, key=lambda item: item[0])
        visible = 1.0
        count_norm = self._clip01(float(len(enemies)) / 8.0)
        return visible, count_norm, largest[1], largest[2], largest[0]

    def _read_state_vector(self, state: Optional[Any], *, update_last: bool) -> np.ndarray:
        now = self._state_vars_snapshot()
        prev = self._last_state_vars or now
        enemy_visible, enemy_count, enemy_cx, enemy_cy, enemy_area = self._enemy_state_features(state)

        damage_delta = max(0.0, now["damage"] - prev.get("damage", now["damage"]))
        hit_delta = max(0.0, now["hits"] - prev.get("hits", now["hits"]))
        hits_taken_delta = max(0.0, now["hits_taken"] - prev.get("hits_taken", now["hits_taken"]))
        frag_delta = now["frags"] - prev.get("frags", now["frags"])
        death_delta = max(0.0, now["deaths"] - prev.get("deaths", now["deaths"]))

        vec = np.asarray(
            [
                self._clip01(now["health"] / 100.0),
                self._clip01(now["armor"] / 100.0),
                self._clip01(now["ammo2"] / 50.0),
                self._clip01(now["selected_weapon"] / 10.0),
                self._clip01(now["selected_weapon_ammo"] / 50.0),
                self._clip01(now["speed_xy"] / 20.0),
                self._clip11(now["vx"] / 20.0),
                self._clip11(now["vy"] / 20.0),
                self._clip01(damage_delta / 100.0),
                self._clip01(hit_delta / 10.0),
                self._clip01(hits_taken_delta / 10.0),
                self._clip11(frag_delta),
                self._clip01(death_delta),
                enemy_visible,
                enemy_count,
                enemy_cx,
                enemy_cy,
                enemy_area,
            ][: self.STATE_DIM],
            dtype=np.float32,
        )

        if update_last:
            self._last_state_vars = now
        return vec

    def _read_observation(self, state: Optional[Any] = None, *, update_state: bool = True) -> Dict[str, np.ndarray]:
        return {
            "image": self._read_obs(state),
            "state": self._read_state_vector(state, update_last=update_state),
        }

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
            raise RuntimeError(f"[ENV] reset(): game nao esta rodando (is_host={self.is_host}, name={self.name})")

        if self._engine_episode_count == 0:
            self._engine_episode_count = 1
            print(
                f"[ENV] reset(): usando episodio criado por init() "
                f"(engine_episode_count={self._engine_episode_count}, is_host={self.is_host}, name={self.name})"
            )
        else:
            print(
                f"[ENV] reset(): reusando episodio existente "
                f"(engine_episode_count={self._engine_episode_count}, is_host={self.is_host}, name={self.name})"
            )

        vars_snapshot = capture_vars(g)
        self.shaper.reset(snapshot=vars_snapshot)
        self.contextual_shaper.reset()
        obs = self._read_observation(update_state=True)
        self._show_agent_view_window(obs["image"])

        print(
            f"[ENV] reset() concluido (is_host={self.is_host}, name={self.name}, "
            f"engine_episode_count={self._engine_episode_count}, "
            f"image_shape={obs['image'].shape}, state_shape={obs['state'].shape})"
        )
        return obs, {"name": self.name}

    def step(self, action: int):
        try:
            if self.game.is_player_dead():
                self.game.respawn_player()

            a = self._actions[int(action)].tolist()
            wall_penalty_enabled = float(getattr(self.agent.reward.shaping, "wall_stuck_penalty", 0.0)) < 0.0
            enemy_reward_enabled = self._labels_enabled and float(
                getattr(self.agent.reward.shaping, "enemy_in_view_reward", 0.0)
            ) > 0.0

            motion_before = self._motion_snapshot() if wall_penalty_enabled else None
            engine_r = float(self.game.make_action(a, int(self.dm.frame_skip)))
            state_after_action = self.game.get_state() if self._labels_enabled else None
            vars_snapshot: StepVarsSnapshot = capture_vars(self.game)
            shaped_r = self.shaper.compute(self.game, engine_r, snapshot=vars_snapshot)
            velocity_snapshot = self._state_vars_snapshot()
            shaped_r += self.contextual_shaper.compute_velocity_reward(
                speed_xy=velocity_snapshot["speed_xy"],
                move_attempt=ContextualRewardShaper._move_attempt(a),
            )

            if wall_penalty_enabled and motion_before is not None:
                motion_after = self._motion_snapshot()
                shaped_r += self.contextual_shaper.compute_wall_stuck(a, motion_before, motion_after)

            if enemy_reward_enabled:
                shaped_r += self.contextual_shaper.compute_enemy_acquisition(
                    state_after_action,
                    screen_w=self.NET_W,
                    screen_h=self.NET_H,
                )

            if self.game.is_episode_finished():
                print(
                    f"[ENV][DEBUG] is_episode_finished()==True em step() "
                    f"(engine_episode_count={self._engine_episode_count}, is_host={self.is_host}, name={self.name})"
                )
                t0 = time.monotonic()
                self.game.new_episode()
                self._engine_episode_count += 1
                self.shaper.reset(self.game)
                self.contextual_shaper.reset()
                elapsed = time.monotonic() - t0
                print(
                    f"[ENV][DEBUG] new_episode() OK em {elapsed:.3f}s "
                    f"(engine_episode_count={self._engine_episode_count}, is_host={self.is_host}, name={self.name})"
                )
                obs = self._read_observation(update_state=True)
            else:
                obs = self._read_observation(state_after_action, update_state=True)
            self._show_agent_view_window(obs["image"])

            info: Dict[str, Any] = {
                "engine_r": engine_r,
                "engine_episode_count": self._engine_episode_count,
            }

            terminated = False
            truncated = False
            return obs, shaped_r, terminated, truncated, info

        except Exception as e:
            print(
                f"[ENV][ERROR] Excecao em step(action={action}) "
                f"(is_host={self.is_host}, name={self.name}): {e!r}"
            )
            raise
    def render(self):
        pass

    def close(self):
        print(f"[ENV] close() chamado (is_host={self.is_host}, name={self.name})")
        if self._render_agent_view:
            try:
                cv2.destroyWindow(self._agent_view_window_name)
            except Exception:
                pass
        try:
            self.game.close()
        except Exception as e:
            print(
                f"[ENV][WARN] Exceção em game.close() "
                f"(is_host={self.is_host}, name={self.name}): {e!r}"
            )

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple
import vizdoom as vzd

from .config import ShapingConfig, EngineRewardConfig
from .config import WallStuckConfig, EnemyInViewConfig

# Variáveis usadas no shaping
VAR_LIST = [
    vzd.GameVariable.FRAGCOUNT,
    vzd.GameVariable.HEALTH,
    vzd.GameVariable.ARMOR,
    vzd.GameVariable.AMMO2,
    vzd.GameVariable.DAMAGECOUNT,
    vzd.GameVariable.DEATHCOUNT,
    vzd.GameVariable.HITCOUNT,
    vzd.GameVariable.HITS_TAKEN,
]


@dataclass
class LastVars:
    vals: Dict[int, int]


@dataclass(frozen=True)
class StepVarsSnapshot:
    vals: Dict[int, int]


@dataclass(frozen=True)
class MotionSnapshot:
    x: float
    y: float
    angle_deg: float

    def as_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.angle_deg)


def capture_vars(game: vzd.DoomGame) -> StepVarsSnapshot:
    return StepVarsSnapshot(vals={int(v): int(game.get_game_variable(v)) for v in VAR_LIST})


def read_vars(game: vzd.DoomGame) -> Dict[int, int]:
    return capture_vars(game).vals


def _term(delta: int, w: float) -> float:
    """
    Função de sinal bem-definido:

    - Se w > 0: só considera delta > 0 (ganhos).
    - Se w < 0: só considera delta < 0 (perdas).
    - Se w == 0: não contribui.

    Assim, o sinal da contribuição é sempre o mesmo sinal do peso.
    Logo, se TODOS os pesos forem >= 0 e step_penalty >= 0,
    o shaping nunca será negativo (ignorando engine reward).
    """
    if w == 0.0:
        return 0.0
    if w > 0.0:
        return w * max(delta, 0)
    else:
        return w * min(delta, 0)


def _count_increase_term(delta: int, w: float) -> float:
    """
    Applies a signed weight to counters where the bad/good event increases.

    Examples:
    - deaths + w_deaths=-5.0 -> -5.0
    - hits_taken + w_hits_taken=-0.2 -> -0.2
    - hits + w_hits=0.5 -> +0.5
    """
    if w == 0.0:
        return 0.0
    return w * max(delta, 0)


class RewardShaper:
    """
    Reward próprio por deltas + step_penalty.

    IMPORTANTE:
    - include_engine_reward é ignorado (engine reward NÃO é somado).
    - Se todos os pesos forem >= 0 e step_penalty >= 0,
      o reward shaping nunca será negativo.
    """
    def __init__(self, cfg: ShapingConfig):
        self.cfg = cfg
        self._last = LastVars(vals={})

    @staticmethod
    def _resolve_now(
        game: Optional[vzd.DoomGame] = None,
        snapshot: Optional[StepVarsSnapshot] = None,
    ) -> Dict[int, int]:
        if snapshot is not None:
            return dict(snapshot.vals)
        if game is None:
            raise ValueError("RewardShaper requer game ou snapshot.")
        return read_vars(game)

    def reset(
        self,
        game: Optional[vzd.DoomGame] = None,
        *,
        snapshot: Optional[StepVarsSnapshot] = None,
    ) -> None:
        self._last.vals = self._resolve_now(game=game, snapshot=snapshot)

    def compute(
        self,
        game: Optional[vzd.DoomGame],
        engine_reward_last_step: float,
        *,
        snapshot: Optional[StepVarsSnapshot] = None,
    ) -> float:
        now = self._resolve_now(game=game, snapshot=snapshot)
        lv = self._last.vals
        d = {k: now[k] - lv.get(k, 0) for k in now}
        r = 0.0

        # Deltas por variável
        d_frag = d[int(vzd.GameVariable.FRAGCOUNT)]
        d_health = d[int(vzd.GameVariable.HEALTH)]
        d_armor = d[int(vzd.GameVariable.ARMOR)]
        d_ammo2 = d[int(vzd.GameVariable.AMMO2)]
        d_damage_made = d[int(vzd.GameVariable.DAMAGECOUNT)]
        d_deaths = d[int(vzd.GameVariable.DEATHCOUNT)]
        d_hits = d[int(vzd.GameVariable.HITCOUNT)]
        d_hits_taken = d[int(vzd.GameVariable.HITS_TAKEN)]

        # FRAG (ganhos/punições controlados pelo sinal de w_frag)
        r += _term(d_frag, self.cfg.w_frag)

        # HEALTH
        r += _term(d_health, self.cfg.w_health)

        # ARMOR
        r += _term(d_armor, self.cfg.w_armor)

        # AMMO2 (custo só quando diminui; sinal controlado por w_ammo2_cost)
        if d_ammo2 < 0:
            r += self.cfg.w_ammo2_cost * (-d_ammo2)

        # DAMAGE DONE
        r += _term(d_damage_made, self.cfg.w_damage_made)

        # HITS: counter increases on successful hits.
        r += _count_increase_term(d_hits, self.cfg.w_hits)

        # HITS_TAKEN: counter increases when the agent is hit.
        r += _count_increase_term(d_hits_taken, self.cfg.w_hits_taken)

        # DEATHS: counter increases when the agent dies.
        r += _count_increase_term(d_deaths, self.cfg.w_deaths)

        # STEP PENALTY (pode ser negativo se vc quiser)
        r += self.cfg.step_penalty

        shaped_final = float(r)

        # LOG DE DEBUG – ainda mostramos engine_r só para inspeção
        

        self._last.vals = now
        return shaped_final


class ContextualRewardShaper:
    """
    Shaping contextual leve para sinais dependentes da acao e do estado atual.

    Mantem memoria curta para:
    - penalidade de persistencia ao ficar preso tentando andar;
    - bonus de aquisicao quando um inimigo entra no campo de visao.
    """

    def __init__(
        self,
        shaping_cfg: ShapingConfig,
        wall_stuck_cfg: WallStuckConfig,
        enemy_in_view_cfg: EnemyInViewConfig,
    ):
        self.shaping_cfg = shaping_cfg
        self.wall_stuck_cfg = wall_stuck_cfg
        self.enemy_in_view_cfg = enemy_in_view_cfg
        self._wall_stuck_streak = 0
        self._enemy_visible_prev = False
        self._enemy_check_counter = 0
        self._enemy_in_view_cooldown = 0

    def reset(self) -> None:
        self._wall_stuck_streak = 0
        self._enemy_visible_prev = False
        self._enemy_check_counter = 0
        self._enemy_in_view_cooldown = 0

    @staticmethod
    def _move_attempt(action_buttons: Iterable[int]) -> bool:
        buttons = list(action_buttons)
        return bool(len(buttons) >= 4 and (buttons[0] or buttons[1] or buttons[2] or buttons[3]))

    @staticmethod
    def _distance_2d(before: Tuple[float, float, float], after: Tuple[float, float, float]) -> float:
        dx = after[0] - before[0]
        dy = after[1] - before[1]
        return float((dx * dx + dy * dy) ** 0.5)

    @staticmethod
    def _angle_delta_deg(before_deg: float, after_deg: float) -> float:
        delta = (after_deg - before_deg + 180.0) % 360.0 - 180.0
        return abs(delta)

    def compute_wall_stuck(
        self,
        action_buttons: Iterable[int],
        before: Tuple[float, float, float] | MotionSnapshot,
        after: Tuple[float, float, float] | MotionSnapshot,
    ) -> float:
        penalty = float(getattr(self.shaping_cfg, "wall_stuck_penalty", 0.0))
        if penalty >= 0.0:
            self._wall_stuck_streak = 0
            return 0.0

        if not self._move_attempt(action_buttons):
            self._wall_stuck_streak = 0
            return 0.0

        before_tuple = before.as_tuple() if isinstance(before, MotionSnapshot) else before
        after_tuple = after.as_tuple() if isinstance(after, MotionSnapshot) else after
        dist = self._distance_2d(before_tuple, after_tuple)
        angle_delta = self._angle_delta_deg(before_tuple[2], after_tuple[2])
        min_move = float(getattr(self.wall_stuck_cfg, "min_move", 1.0))
        max_turn_deg = float(getattr(self.wall_stuck_cfg, "max_turn_deg", 5.0))
        persist_steps = max(1, int(getattr(self.wall_stuck_cfg, "persist_steps", 4)))

        stuck_candidate = dist < min_move and angle_delta < max_turn_deg
        if stuck_candidate:
            self._wall_stuck_streak += 1
        else:
            self._wall_stuck_streak = 0

        if self._wall_stuck_streak >= persist_steps:
            return penalty
        return 0.0

    def compute_enemy_acquisition(
        self,
        state: Optional[Any],
        *,
        screen_w: int,
        screen_h: int,
    ) -> float:
        reward = float(getattr(self.shaping_cfg, "enemy_in_view_reward", 0.0))
        if reward <= 0.0:
            return 0.0

        if self._enemy_in_view_cooldown > 0:
            self._enemy_in_view_cooldown -= 1

        check_every = max(1, int(getattr(self.enemy_in_view_cfg, "check_every", 2)))
        self._enemy_check_counter += 1
        if self._enemy_check_counter % check_every != 0:
            return 0.0

        enemy_visible_now = self._has_enemy_label(
            state,
            screen_w=screen_w,
            screen_h=screen_h,
            min_area_ratio=float(getattr(self.enemy_in_view_cfg, "min_area_ratio", 0.002)),
        )

        should_reward = (
            enemy_visible_now
            and not self._enemy_visible_prev
            and self._enemy_in_view_cooldown == 0
        )

        self._enemy_visible_prev = enemy_visible_now
        if should_reward:
            self._enemy_in_view_cooldown = max(
                0, int(getattr(self.enemy_in_view_cfg, "cooldown_steps", 3))
            )
            return reward
        return 0.0

    def compute_velocity_reward(self, *, speed_xy: float, move_attempt: bool) -> float:
        speed_weight = float(getattr(self.shaping_cfg, "w_speed", 0.0))
        idle_threshold = max(0.0, float(getattr(self.shaping_cfg, "idle_speed_threshold", 0.0)))
        idle_penalty = float(getattr(self.shaping_cfg, "idle_penalty", 0.0))

        speed_norm = min(max(float(speed_xy), 0.0), 20.0) / 20.0
        reward = speed_weight * speed_norm
        if move_attempt and idle_penalty < 0.0 and speed_norm < idle_threshold:
            reward += idle_penalty
        return float(reward)

    @staticmethod
    def _has_enemy_label(
        state: Optional[Any],
        *,
        screen_w: int,
        screen_h: int,
        min_area_ratio: float,
    ) -> bool:
        if state is None:
            return False

        labels = getattr(state, "labels", None)
        if not labels:
            return False

        denom = max(1.0, float(screen_w * screen_h))
        for label in labels:
            category = str(getattr(label, "object_category", "") or "")
            if category != "Player":
                continue

            width = max(0.0, float(getattr(label, "width", 0.0) or 0.0))
            height = max(0.0, float(getattr(label, "height", 0.0) or 0.0))
            if (width * height) / denom >= min_area_ratio:
                return True

        return False


def apply_engine_rewards(game: vzd.DoomGame, ercfg: EngineRewardConfig) -> None:
    """
    Configura recompensas nativas do ViZDoom (se suportadas na sua versão).

    Mesmo que o RewardShaper NÃO some o engine reward,
    ainda é útil setar isso para testes e para quem quer usar
    apenas EngineReward no futuro.
    """
    if hasattr(game, "set_living_reward"):
        game.set_living_reward(ercfg.living_reward)
    for meth, val in [
        ("set_frag_reward", ercfg.frag_reward),
        ("set_hit_reward", ercfg.hit_reward),
        ("set_hit_taken_reward", ercfg.hit_taken_reward),
        ("set_damage_made_reward", ercfg.damage_made_reward),
        ("set_damage_taken_penalty", ercfg.damage_taken_penalty),
        ("set_item_reward", ercfg.item_reward),
        ("set_health_reward", ercfg.health_reward),
        ("set_armor_reward", ercfg.armor_reward),
        ("set_secret_reward", ercfg.secret_reward),
    ]:
        if hasattr(game, meth):
            getattr(game, meth)(val)

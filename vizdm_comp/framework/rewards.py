from dataclasses import dataclass
from typing import Dict
import vizdoom as vzd

# Variáveis usadas no shaping
VAR_LIST = [
    vzd.GameVariable.FRAGCOUNT,
    vzd.GameVariable.HEALTH,
    vzd.GameVariable.ARMOR,
    vzd.GameVariable.AMMO2,
    vzd.GameVariable.DEATHCOUNT,
    vzd.GameVariable.HITCOUNT,
    vzd.GameVariable.HITS_TAKEN,
]

@dataclass
class LastVars:
    vals: Dict[int, int]

def read_vars(game: vzd.DoomGame) -> Dict[int, int]:
    return {int(v): int(game.get_game_variable(v)) for v in VAR_LIST}

class RewardShaper:
    """
    Reward próprio por deltas + step_penalty (+ opcional engine_reward).
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self._last = LastVars(vals={})

    def reset(self, game: vzd.DoomGame) -> None:
        self._last.vals = read_vars(game)

    def compute(self, game: vzd.DoomGame, engine_reward_last_step: float) -> float:
        now = read_vars(game)
        lv = self._last.vals
        d = {k: now[k] - lv.get(k, 0) for k in now}
        r = 0.0
        r += self.cfg.w_frag       * d[int(vzd.GameVariable.FRAGCOUNT)]
        r += self.cfg.w_health     * d[int(vzd.GameVariable.HEALTH)]
        r += self.cfg.w_armor      * d[int(vzd.GameVariable.ARMOR)]
        ammo_delta = d[int(vzd.GameVariable.AMMO2)]
        if ammo_delta < 0:
            r += self.cfg.w_ammo2_cost * (-ammo_delta)
        r += self.cfg.w_hits       * d[int(vzd.GameVariable.HITCOUNT)]
        r += self.cfg.w_hits_taken * d[int(vzd.GameVariable.HITS_TAKEN)]
        r += self.cfg.w_deaths     * d[int(vzd.GameVariable.DEATHCOUNT)]
        r += self.cfg.step_penalty
        if self.cfg.include_engine_reward:
            r += engine_reward_last_step
        self._last.vals = now
        return float(r)

def apply_engine_rewards(game: vzd.DoomGame, ercfg) -> None:
    """
    Configura recompensas nativas do ViZDoom (se suportadas na sua versão).
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

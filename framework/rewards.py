from dataclasses import dataclass
from typing import Dict
import vizdoom as vzd

from .config import ShapingConfig, EngineRewardConfig

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

    def reset(self, game: vzd.DoomGame) -> None:
        self._last.vals = read_vars(game)

    def compute(self, game: vzd.DoomGame, engine_reward_last_step: float) -> float:
        now = read_vars(game)
        lv = self._last.vals
        d = {k: now[k] - lv.get(k, 0) for k in now}
        r = 0.0

        # Deltas por variável
        d_frag = d[int(vzd.GameVariable.FRAGCOUNT)]
        d_health = d[int(vzd.GameVariable.HEALTH)]
        d_armor = d[int(vzd.GameVariable.ARMOR)]
        d_ammo2 = d[int(vzd.GameVariable.AMMO2)]
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

        # HITS
        r += _term(d_hits, self.cfg.w_hits)

        # HITS_TAKEN
        r += _term(d_hits_taken, self.cfg.w_hits_taken)

        # DEATHS
        r += _term(d_deaths, self.cfg.w_deaths)

        # STEP PENALTY (pode ser negativo se vc quiser)
        r += self.cfg.step_penalty

        shaped_final = float(r)

        # LOG DE DEBUG – ainda mostramos engine_r só para inspeção
        

        self._last.vals = now
        return shaped_final


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

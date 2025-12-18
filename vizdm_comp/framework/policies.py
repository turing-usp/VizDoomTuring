from typing import Dict, Any, Optional
import importlib
import torch
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.base_class import BaseAlgorithm

AlgoMap = {"ppo": PPO, "a2c": A2C, "dqn": DQN}

def resolve_algo(algo_name: str):
    algo_name = algo_name.lower()
    if algo_name not in AlgoMap:
        raise ValueError(f"Algo '{algo_name}' não suportado.")
    return AlgoMap[algo_name]

def maybe_import_class(path_and_cls: Optional[str]):
    """
    Carrega classe "pkg.mod:ClassName" dinamicamente (extratores/adapters).
    """
    if not path_and_cls:
        return None
    mod_path, cls_name = path_and_cls.split(":")
    mod = importlib.import_module(mod_path)
    return getattr(mod, cls_name)

class ExternalPolicyAdapter:
    """
    Adapter p/ política externa (PyTorch puro) com método .predict(obs).
    Espera policy_kwargs["external_class"] = "mymod:MyNet".
    """
    def __init__(self, weights_path: str, n_actions: int, external_class: str):
        cls = maybe_import_class(external_class)
        if cls is None:
            raise ValueError("external_class ausente (use 'mymod:MyNet').")
        self.net = cls(n_actions=n_actions)
        self.net.load_state_dict(torch.load(weights_path, map_location="cpu"))
        self.net.eval()

    @torch.no_grad()
    def predict(self, obs, deterministic: bool = True):
        # obs: (N,C,H,W) uint8
        x = torch.from_numpy(obs).float() / 255.0
        logits = self.net(x)              # N x n_actions
        act = torch.argmax(logits, dim=1).cpu().numpy()
        return act, None

def _coerce_learn_kwargs(learn_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Converte strings numéricas do YAML para float/int (ex.: "3e-4" -> 0.0003),
    evitando AssertionError do SB3.
    """
    lk = dict(learn_kwargs)
    float_keys = ["learning_rate", "gamma", "gae_lambda", "ent_coef", "vf_coef", "clip_range", "target_kl"]
    int_keys   = ["n_steps", "batch_size", "n_epochs", "train_freq", "target_update_interval", "buffer_size"]
    for k in float_keys:
        if k in lk and isinstance(lk[k], str):
            lk[k] = float(lk[k])
    for k in int_keys:
        if k in lk and isinstance(lk[k], str):
            lk[k] = int(lk[k])
    return lk

def build_sb3(
    algo_cls, policy_str: str, env, policy_kwargs: Dict[str, Any], learn_kwargs: Dict[str, Any]
) -> BaseAlgorithm:
    """
    Instancia algoritmo SB3 com import dinâmico do features_extractor se preciso.
    """
    # import dinâmico para extrator custom
    fekey = "features_extractor_class"
    if fekey in policy_kwargs and isinstance(policy_kwargs[fekey], str):
        policy_kwargs = dict(policy_kwargs)
        policy_kwargs[fekey] = maybe_import_class(policy_kwargs[fekey])

    # normaliza tipos numéricos de learn_kwargs
    learn_kwargs = _coerce_learn_kwargs(learn_kwargs)

    model = algo_cls(policy_str, env, verbose=1, device="auto",
                     policy_kwargs=policy_kwargs, **learn_kwargs)
    return model

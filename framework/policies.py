from __future__ import annotations

from typing import Dict, Any, Optional, Type
import importlib

import torch
from gymnasium import spaces
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.base_class import BaseAlgorithm

from .ppo_cuda import CudaOptimizedPPO
from .rollout_buffers import PinnedRolloutBuffer

# ---------------------------------------------------------
# Mapa de algoritmos suportados
# ---------------------------------------------------------

AlgoMap = {
    "ppo": PPO,
    "a2c": A2C,
    "dqn": DQN,
}

_LEARNER_RUNTIME_KEYS = {
    "use_pinned_rollout_buffer",
    "use_cuda_optimized_ppo",
    "use_amp",
    "use_torch_compile",
    "use_cudnn_benchmark",
    "use_tf32",
    "torch_matmul_precision",
}


def resolve_algo(algo_name: str):
    """
    Resolve nome de algoritmo ('ppo' | 'a2c' | 'dqn') para a classe SB3 correspondente.
    """
    algo_name = algo_name.lower()
    if algo_name not in AlgoMap:
        raise ValueError(f"Algo '{algo_name}' não suportado.")
    return AlgoMap[algo_name]


def resolve_effective_algo(algo_name: str, learn_kwargs: Optional[Dict[str, Any]] = None):
    """Resolve the class used for load/create, including optional CUDA PPO."""
    algo_cls = resolve_algo(algo_name)
    lk = learn_kwargs or {}
    if algo_name.lower() == "ppo" and bool(lk.get("use_cuda_optimized_ppo", False)):
        return CudaOptimizedPPO
    return algo_cls


# ---------------------------------------------------------
# Import dinâmico de classes externas
# ---------------------------------------------------------


def maybe_import_class(path_and_cls: Optional[str]):
    """
    Carrega classe "pkg.mod:ClassName" dinamicamente (extratores/adapters).

    Exemplo:
        "my_package.my_module:MyNet" -> my_package.my_module.MyNet
    """
    if not path_and_cls:
        return None
    if ":" not in path_and_cls:
        raise ValueError(f"Formato inválido para classe externa: {path_and_cls!r}")
    mod_path, cls_name = path_and_cls.split(":", 1)
    mod = importlib.import_module(mod_path)
    return getattr(mod, cls_name)


# ---------------------------------------------------------
# Adapter para política externa (PyTorch puro)
# ---------------------------------------------------------


class ExternalPolicyAdapter:
    """
    Adapter p/ política externa (PyTorch puro) com método .predict(obs).

    Espera que policy_kwargs["external_class"] contenha o caminho da classe:
        "mymod:MyNet"

    A rede externa recebe obs em float32 normalizado (0..1), shape (N, C, H, W),
    e deve retornar logits (N, n_actions).
    """

    def __init__(self, weights_path: str, n_actions: int, external_class: str):
        cls = maybe_import_class(external_class)
        if cls is None:
            raise ValueError("external_class ausente (use 'mymod:MyNet').")

        self.net = cls(n_actions=n_actions)
        # carrega pesos sempre na CPU (adaptador simples)
        state = torch.load(weights_path, map_location="cpu")
        self.net.load_state_dict(state)
        self.net.eval()

    @torch.no_grad()
    def predict(self, obs, deterministic: bool = True):
        """
        obs: np.ndarray (N, C, H, W), dtype uint8
        Retorna: (actions, None), onde actions é np.ndarray (N,)
        """
        x = torch.from_numpy(obs).float() / 255.0
        logits = self.net(x)              # N x n_actions
        act = torch.argmax(logits, dim=1).cpu().numpy()
        return act, None


# ---------------------------------------------------------
# Normalização de learn_kwargs vindos do YAML
# ---------------------------------------------------------


def _coerce_learn_kwargs(learn_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Converte strings numéricas do YAML para float/int
    (ex.: "3e-4" -> 0.0003), evitando AssertionError do SB3.
    """
    lk = dict(learn_kwargs)

    float_keys = [
        "learning_rate",
        "learning_rate_max",
        "learning_rate_min",
        "ent_coef_max",
        "ent_coef_min",
        "gamma",
        "gae_lambda",
        "ent_coef",
        "vf_coef",
        "clip_range",
        "target_kl",
    ]
    int_keys = [
        "n_steps",
        "batch_size",
        "n_epochs",
        "train_freq",
        "target_update_interval",
        "buffer_size",
        "n_envs",
    ]
    bool_keys = [
        "use_pinned_rollout_buffer",
        "use_cuda_optimized_ppo",
        "use_amp",
        "use_torch_compile",
        "use_cudnn_benchmark",
        "use_tf32",
    ]

    for k in float_keys:
        if k in lk and isinstance(lk[k], str):
            lk[k] = float(lk[k])
    for k in int_keys:
        if k in lk and isinstance(lk[k], str):
            lk[k] = int(lk[k])
    for k in bool_keys:
        if k in lk and isinstance(lk[k], str):
            lk[k] = lk[k].strip().lower() in {"1", "true", "yes", "y", "on"}

    return lk


def _extract_learner_runtime_kwargs(learn_kwargs: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    lk = dict(learn_kwargs)
    runtime = {key: lk.pop(key) for key in list(lk.keys()) if key in _LEARNER_RUNTIME_KEYS}
    return lk, runtime


def _maybe_enable_cuda_learner(
    algo_cls: Type[BaseAlgorithm],
    learn_kwargs: Dict[str, Any],
    runtime_kwargs: Dict[str, Any],
    device: str,
) -> tuple[Type[BaseAlgorithm], Dict[str, Any]]:
    if algo_cls is PPO and bool(runtime_kwargs.get("use_cuda_optimized_ppo", False)):
        algo_cls = CudaOptimizedPPO

    if algo_cls is CudaOptimizedPPO:
        learn_kwargs["use_amp"] = bool(runtime_kwargs.get("use_amp", False))
        learn_kwargs["compile_evaluate_actions"] = bool(runtime_kwargs.get("use_torch_compile", False))

    if algo_cls in {PPO, CudaOptimizedPPO} and bool(runtime_kwargs.get("use_pinned_rollout_buffer", False)):
        learn_kwargs["rollout_buffer_class"] = PinnedRolloutBuffer
        learn_kwargs["rollout_buffer_kwargs"] = {
            **dict(learn_kwargs.get("rollout_buffer_kwargs") or {}),
            "pin_memory_if_cuda": device == "cuda",
        }

    return algo_cls, learn_kwargs


# ---------------------------------------------------------
# Fábrica de modelos SB3 (PPO/A2C/DQN) com seleção de device
# ---------------------------------------------------------


def build_sb3(
    algo_cls,
    policy_str: str,
    env,
    policy_kwargs: Dict[str, Any],
    learn_kwargs: Dict[str, Any],
) -> BaseAlgorithm:
    """
    Instancia algoritmo SB3 com import dinâmico do features_extractor se preciso,
    normaliza learn_kwargs e escolhe device (cuda se disponível, senão cpu).

    Suporte a scheduler externo:
    - Se learn_kwargs contiver learning_rate_max e learning_rate_min, eles são
      removidos dos kwargs passados ao SB3 e armazenados em model._lr_range.
    - Se não houver 'learning_rate' explícito, usamos learning_rate_max como
      valor inicial.
    """
    # import dinâmico para extrator custom (features_extractor_class)
    fekey = "features_extractor_class"
    if fekey in policy_kwargs and isinstance(policy_kwargs[fekey], str):
        policy_kwargs = dict(policy_kwargs)
        policy_kwargs[fekey] = maybe_import_class(policy_kwargs[fekey])

    # normaliza tipos numéricos de learn_kwargs
    learn_kwargs = _coerce_learn_kwargs(learn_kwargs)
    learn_kwargs, runtime_kwargs = _extract_learner_runtime_kwargs(learn_kwargs)

    # extrai range de LR, se houver
    lr_max = learn_kwargs.pop("learning_rate_max", None)
    lr_min = learn_kwargs.pop("learning_rate_min", None)
    ent_max = learn_kwargs.pop("ent_coef_max", None)
    ent_min = learn_kwargs.pop("ent_coef_min", None)

    # se não há learning_rate explícito e temos lr_max, usamos ele como base
    if "learning_rate" not in learn_kwargs and lr_max is not None:
        learn_kwargs["learning_rate"] = lr_max
    if "ent_coef" not in learn_kwargs and ent_max is not None:
        learn_kwargs["ent_coef"] = ent_max

    # escolhe device explicitamente
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    print(f"[POLICY] device={device}, cuda_available={use_cuda}")

    if isinstance(getattr(env, "observation_space", None), spaces.Dict):
        if policy_str == "CnnPolicy":
            policy_str = "MultiInputPolicy"
            print("[POLICY] Observation Dict detectado; usando MultiInputPolicy.")
        if bool(runtime_kwargs.get("use_pinned_rollout_buffer", False)):
            runtime_kwargs["use_pinned_rollout_buffer"] = False
            print("[POLICY][WARN] PinnedRolloutBuffer desativado para observation_space Dict.")

    algo_cls, learn_kwargs = _maybe_enable_cuda_learner(
        algo_cls,
        learn_kwargs,
        runtime_kwargs,
        device,
    )
    if algo_cls is CudaOptimizedPPO:
        print(
            "[POLICY] CudaOptimizedPPO ativo "
            f"(amp={bool(runtime_kwargs.get('use_amp', False))}, "
            f"compile={bool(runtime_kwargs.get('use_torch_compile', False))})"
        )
    if learn_kwargs.get("rollout_buffer_class") is PinnedRolloutBuffer:
        print("[POLICY] PinnedRolloutBuffer ativo para PPO.")

    model = algo_cls(
        policy_str,
        env,
        verbose=1,
        device=device,
        policy_kwargs=policy_kwargs,
        **learn_kwargs,
    )

    # salva range de LR no modelo para o scheduler externo
    if lr_max is not None and lr_min is not None:
        try:
            model._lr_range = (float(lr_max), float(lr_min))
            print(
                f"[POLICY] LR range configurado em model._lr_range = "
                f"({float(lr_max)}, {float(lr_min)})"
            )
        except Exception as e:
            print(f"[POLICY][WARN] Não foi possível setar _lr_range: {e}")

    # sanity-check rápido
    if ent_max is not None and ent_min is not None:
        try:
            model._ent_coef_range = (float(ent_max), float(ent_min))
            print(
                f"[POLICY] Entropy range configurado em model._ent_coef_range = "
                f"({float(ent_max)}, {float(ent_min)})"
            )
        except Exception as e:
            print(f"[POLICY][WARN] NÃ£o foi possÃ­vel setar _ent_coef_range: {e}")

    print(f"[POLICY] SB3 model.device = {getattr(model, 'device', 'unknown')}")

    return model

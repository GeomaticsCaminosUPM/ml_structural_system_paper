"""Single place to check whether a CUDA GPU is available; falls back to CPU."""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def cuda_available() -> bool:
    """Whether a CUDA GPU is visible to torch right now."""
    import torch

    return torch.cuda.is_available()


def xgboost_device() -> str:
    """The ``device`` kwarg XGBoost's ``XGBClassifier`` should be built with."""
    return "cuda" if cuda_available() else "cpu"


def catboost_task_type() -> str:
    """The ``task_type`` kwarg CatBoost's ``CatBoostClassifier`` should be built with.
    """
    return "GPU" if cuda_available() else "CPU"


from xgboost import XGBClassifier  # noqa: E402, already a hard dependency of this package


class XGBClassifierCPUPredict(XGBClassifier):
    """Drop-in replacement for ``XGBClassifier``: trains on whatever ``device`` it's
    constructed with (``xgboost_device()``, i.e.
    """

    def fit(self, *args, **kwargs):
        super().fit(*args, **kwargs)
        if self.get_params().get("device") == "cuda":
            self.set_params(device="cpu")
        return self

"""ml_structural_system, public API."""

from typing import Any

from .config import BootstrapConfig, ExperimentConfig, load_config
from .exceptions import ConfigError, MLStructuralSystemError, ModelNotRegisteredError
from .assess import (
    AssessmentResult,
    BootstrapCI,
    GlobalAgreementStats,
    ModelGeneralizabilityStats,
    ModelInferenceResult,
    SampleAgreementStats,
    SoftEnsembleResult,
    run as assess_run,
)

# Submodules loaded on first access (see __getattr__).
_LAZY_SUBMODULES = frozenset(
    {"train", "test", "assess", "models", "cli"}
)


def __getattr__(name: str) -> Any:
    """Lazily import and cache a submodule accessed as a package attribute."""
    if name in _LAZY_SUBMODULES:
        import importlib
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module  # cache on the package so re-access is free
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """List public attributes and lazy submodules for introspection/autocomplete."""
    return sorted(set(__all__) | _LAZY_SUBMODULES)


__all__ = [
    # Config
    "ExperimentConfig",
    "BootstrapConfig",
    "load_config",
    # Exceptions
    "MLStructuralSystemError",
    "ConfigError",
    "ModelNotRegisteredError",
    # Assessment
    "AssessmentResult",
    "BootstrapCI",
    "GlobalAgreementStats",
    "ModelGeneralizabilityStats",
    "ModelInferenceResult",
    "SampleAgreementStats",
    "SoftEnsembleResult",
    "assess_run",
    # Lazy submodules
    "train",
    "test",
    "models",
    "cli",
]

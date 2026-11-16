"""Project-specific exceptions."""

from __future__ import annotations


class MLStructuralSystemError(Exception):
    """Base class for all errors raised by this package."""


class ModelNotRegisteredError(MLStructuralSystemError):
    """Raised when a model name in config.yaml has no matching pipeline class."""


class ConfigError(MLStructuralSystemError):
    """Raised when config.yaml is missing a required value or points to a bad path."""


class DatasetError(MLStructuralSystemError):
    """Raised when raw or new data fails a preparation/validation check."""

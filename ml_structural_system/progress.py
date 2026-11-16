"""Disable tqdm bars when output is not a terminal, so redirected logs stay short."""

from __future__ import annotations

import sys


def bar_disabled() -> bool:
    """Whether tqdm progress bars should be disabled (no real terminal attached)."""
    return not sys.stderr.isatty()

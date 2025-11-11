"""Utility helpers for interacting with Weights & Biases.

This module centralizes all optional wandb interactions so the rest of the
codebase can freely depend on these helpers without importing wandb directly.
"""

from __future__ import annotations

from typing import Any, Mapping, MutableMapping, Optional

try:  # pragma: no cover - wandb is optional at runtime
    import wandb as _wandb
except Exception:  # pragma: no cover - fallback when wandb is missing
    _wandb = None  # type: ignore


def _resolve_run(run: Optional[Any]) -> Optional[Any]:
    """Return the provided run or the global wandb run if available."""

    if run is not None:
        return run
    if _wandb is not None:
        return _wandb.run
    return None


def login(*args: Any, **kwargs: Any) -> bool:
    """Proxy for ``wandb.login`` that safely handles missing installs."""

    if _wandb is None:
        return False
    try:
        _wandb.login(*args, **kwargs)
        return True
    except Exception:
        return False


def init_run(*args: Any, **kwargs: Any) -> Optional[Any]:
    """Proxy for ``wandb.init`` returning ``None`` when wandb is unavailable."""

    if _wandb is None:
        return None
    return _wandb.init(*args, **kwargs)


def finish_run(run: Optional[Any] = None) -> None:
    """Finish a run created via :func:`init_run` if possible."""

    resolved = _resolve_run(run)
    if resolved is not None:
        resolved.finish()


def log_data(
    data: Mapping[str, Any] | MutableMapping[str, Any],
    *,
    wandb_run: Optional[Any] = None,
    step: Optional[int] = None,
    commit: Optional[bool] = None,
) -> None:
    """Safely log a payload to wandb if a run is active."""

    if not data:
        return

    resolved = _resolve_run(wandb_run)
    if resolved is None:
        return

    kwargs = {}
    if step is not None:
        kwargs["step"] = step
    if commit is not None:
        kwargs["commit"] = commit

    resolved.log(dict(data), **kwargs)


def create_histogram(values: Any, *, num_bins: int = 512) -> Optional[Any]:
    """Return a wandb Histogram object when the library is installed."""

    if _wandb is None:
        return None
    return _wandb.Histogram(values, num_bins=num_bins)


__all__ = [
    "login",
    "init_run",
    "finish_run",
    "log_data",
    "create_histogram",
]


"""Utility helpers for interacting with Weights & Biases.

This module centralizes all optional wandb interactions so the rest of the
codebase can freely depend on these helpers without importing wandb directly.
"""

from __future__ import annotations

from typing import Any, Mapping, MutableMapping, Optional
import numpy as np
import warnings

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


def _create_histogram(values: Any, *, num_bins: int = 512) -> Optional[Any]:
    """Return a wandb Histogram object when the library is installed."""

    if _wandb is None:
        return None
    return _wandb.Histogram(values, num_bins=num_bins)


def log_model_parameters(model, epoch, total_epochs, logging_frequency=2):
    """
    Logs model parameters, stats, and histograms to W&B.

    This function has special handling for LSTM layers, assuming they follow
    the C++ backend's concatenation order (Forget, Input, Candidate, Output).

    It assumes the model's .state_dict() returns a dictionary where each
    value is a tuple/list of 4 parameters: (mu_w, var_w, mu_b, var_b).

    Args:
        model: The model object (must have a .state_dict() method).
        epoch: The current training epoch number (for x-axis).
        total_epochs: The total number of epochs (to force log on last epoch).
        logging_frequency: How often (in epochs) to log.
    """

    # Check if it's time to log this epoch
    is_last_epoch = epoch == total_epochs - 1
    if (epoch % logging_frequency != 0) and not is_last_epoch:
        return  # Not time to log, skip

    log_payload = {}
    state_dict = model.state_dict()

    for layer_name, params in state_dict.items():
        is_lstm = "lstm" in layer_name.lower()
        gate_names = ("forget", "input", "candidate", "output")
        param_labels = ("mu_w", "var_w", "mu_b", "var_b")

        for param, label in zip(params, param_labels):
            if param is None:
                continue

            values = np.asarray(param)

            # Skip empty parameters (e.g. bias when bias=False)
            if values.size == 0:
                continue

            # Handle NaNs for safe logging
            if np.isnan(values).any():
                values = np.nan_to_num(values, nan=0.0)

            if is_lstm:
                flat_values = values

                # Check divisibility and split for LSTM gates
                if flat_values.size > 0 and flat_values.size % 4 == 0:
                    gate_chunks = np.split(flat_values, 4)

                    for gate_name, gate_data in zip(gate_names, gate_chunks):

                        # params/LSTM.0/mu_w/forget
                        base_key = f"params/{layer_name}/{label}/{gate_name}"

                        # Log histogram using proxy function
                        hist = _create_histogram(gate_data, num_bins=512)
                        if hist is not None:
                            log_payload[f"{base_key}"] = hist

                    continue  # Skip the general histogram logging

                else:
                    warnings.warn(
                        f"LSTM parameter {layer_name}/{label} has size "
                        f"{flat_values.size}, which is not divisible by 4. "
                        "Logging as a single histogram."
                    )

            # Fallback for non-LSTM layers or failed split
            hist_fallback = _create_histogram(values, num_bins=512)
            if hist_fallback is not None:
                log_payload[f"params/{layer_name}/{label}/hist"] = hist_fallback

    return log_payload


__all__ = [
    "login",
    "init_run",
    "finish_run",
    "log_data",
    "create_histogram",
    "log_model_parameters",
]

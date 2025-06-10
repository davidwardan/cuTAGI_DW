"""
view_param.py ― cuTAGI / PyTorch parameter visualiser
-----------------------------------------------------
• Understands TAGI buffers (mu_w / var_w) plus classic .weight tensors.
• Robust to raw Python lists.
• Handles standard Dense/Conv layers and custom cuTAGI LSTM layers.
• Provides:
    - CuTAGIParameterViewer.heatmap(layer_name, which={"mean","var"})
"""

from __future__ import annotations
from pathlib import Path
from typing import Literal, Union

import matplotlib.pyplot as plt
import numpy as np


# ------------------------------------------------------------------ #
#  Utils
# ------------------------------------------------------------------ #
def _to_numpy(x) -> np.ndarray:
    """
    Convert torch.Tensor | np.ndarray | list-like to a NumPy array on CPU.
    (import kept local to avoid a hard torch dependency.)
    """
    if hasattr(x, "detach"):  # torch.Tensor
        return x.detach().cpu().float().numpy()
    return np.asarray(x, dtype=np.float32)


# ------------------------------------------------------------------ #
#  Main viewer class
# ------------------------------------------------------------------ #
class ParameterViewer:
    """
    Quick visualisation helper for cuTAGI (or PyTorch) models.

    Parameters
    ----------
    model      : pytagi.nn.Sequential | torch.nn.Module
    out_dir    : str | Path | None   – if given, save figs there instead of showing
    default_v  : {"mean","var","both"} – default family of params to plot

    Additional:
        • set_epoch(epoch) ― store current training epoch
        • heatmap(..., epoch=None, cmap=None) ― optional epoch stamp + custom cmap
    """

    def __init__(
        self,
        model,
        out_dir: Union[str, Path, None] = "nn_param_plots",
        default_v: Literal["mean", "var", "both"] = "mean",
    ):
        self.model = model
        self.default_v = default_v
        self.out_dir = Path(out_dir) if out_dir else None
        if self.out_dir:
            self.out_dir.mkdir(parents=True, exist_ok=True)
        self.current_epoch: int | None = None

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------
    def heatmap(
        self,
        layer_name: str,
        which: Literal["mean", "var"] = "mean",
        *,
        epoch: int | None = None,
        cmap: str | None = None,
        return_img: bool = False,
        vmin: float | None = None,
        vmax: float | None = None,
    ):
        """
        Visualise the weight matrix / conv filter bank of one layer.

        layer_name – name reported by Sequential/named_modules ("0", "fc1", …)
        which      – "mean" for μ, "var" for σ²
        epoch      – optional int, appended to filename as "ep###"
        cmap       – Matplotlib colour‑map name (e.g. "viridis", "plasma")
        return_img – if True, returns the RGB image as a NumPy array
        vmin, vmax – optional floats to set a fixed color scale for the heatmap
        """
        epoch = int(epoch) if epoch is not None else self.current_epoch
        layer = self._locate_layer(layer_name)
        attr = f"{'mu' if which == 'mean' else 'var'}_w"
        if not hasattr(layer, attr):
            raise ValueError(f"Layer '{layer_name}' has no {attr} buffer.")
        w = _to_numpy(getattr(layer, attr))
        fig = None

        # ---- LSTM Layer: 4 gates in one vector ------------------------
        if (
            w.ndim == 1
            and hasattr(layer, "input_size")
            and hasattr(layer, "output_size")
            and not hasattr(layer, "in_features")
        ):
            out_f, in_f = layer.output_size, layer.input_size + layer.output_size
            gate_size = out_f * in_f

            if w.size == 4 * gate_size:
                gate_names = ["Forget", "Input", "Cell State", "Output"]
                fig, axarr = plt.subplots(2, 2, figsize=(12, 9), tight_layout=True)
                fig.suptitle(
                    f"{which.capitalize()} weights – LSTM Layer '{layer_name}'",
                    fontsize=16,
                )

                vmin_plot = vmin if vmin is not None else np.min(w)
                vmax_plot = vmax if vmax is not None else np.max(w)

                for i, (ax, name) in enumerate(zip(axarr.flat, gate_names)):
                    gate_w = w[i * gate_size : (i + 1) * gate_size].reshape(out_f, in_f)

                    im = ax.imshow(
                        gate_w,
                        aspect="auto",
                        interpolation="nearest",
                        vmin=vmin_plot,
                        vmax=vmax_plot,
                        cmap=cmap,
                    )
                    fig.colorbar(im, ax=ax, shrink=0.8)
                    ax.set_title(f"{name} Gate")
                    ax.set_xlabel(f"Input Weights (left) + Recurrent Weights (right)")
                    ax.set_ylabel(f"Hidden Dim ({layer.output_size})")
                    ax.grid(which="minor", color="w", linestyle="-", linewidth=0.5)
                    ax.tick_params(
                        which="both",
                        bottom=False,
                        left=False,
                        labelbottom=False,
                        labelleft=False,
                    )

                    # --- ADDED FOR CLARITY ---
                    # Draw a vertical line to separate the input weights from the recurrent (hidden) weights.
                    # The first `layer.input_size` columns are for the input, the rest are for the hidden state.
                    split_point = layer.input_size - 0.5
                    ax.axvline(split_point, color="r", linestyle="--", linewidth=2)
                    # --- END ADDED SECTION ---

            else:
                w = w.reshape(1, -1)

        # ---- Convolutional kernels ------------------------------------
        if w.ndim == 4:
            vmin_plot, vmax_plot = vmin, vmax
            if vmin_plot is None and vmax_plot is None:
                vmax_plot = np.max(np.abs(w)) if which == "mean" else np.max(w)
                vmin_plot = -vmax_plot if which == "mean" else np.min(w)

            oc, ic, *_ = w.shape
            fig, axarr = plt.subplots(
                oc, ic, figsize=(ic * 1.4, oc * 1.4), squeeze=False
            )
            for i in range(oc):
                for j in range(ic):
                    axarr[i, j].imshow(
                        w[i, j],
                        cmap=cmap,
                        vmin=vmin_plot,
                        vmax=vmax_plot,
                        interpolation="nearest",
                    )
                    axarr[i, j].axis("off")
            fig.suptitle(f"{which.capitalize()} conv filters – {layer_name}")

        # ---- Dense layer heat-map (or reshaped 1D) --------------------
        elif w.ndim <= 2:
            if w.ndim == 1:
                out_f = None
                for a in ("out_features", "out_dim", "output_features", "output_size"):
                    if hasattr(layer, a):
                        out_f = getattr(layer, a)
                        break
                if out_f is None:
                    for b in (f"{'mu' if which == 'mean' else 'var'}_b", "bias"):
                        if hasattr(layer, b) and getattr(layer, b) is not None:
                            out_f = _to_numpy(getattr(layer, b)).size or None
                            if out_f:
                                break
                w = (
                    w.reshape((out_f, w.size // out_f))
                    if out_f and w.size % out_f == 0
                    else w.reshape(1, -1)
                )

            vmin_plot = vmin if vmin is not None else float(np.min(w))
            vmax_plot = vmax if vmax is not None else float(np.max(w))

            if vmin_plot == vmax_plot:
                eps = 1e-6 or abs(vmin_plot) * 1e-3
                vmin_plot -= eps
                vmax_plot += eps

            fig, ax = plt.subplots()
            im = ax.imshow(
                w,
                aspect="auto",
                interpolation="nearest",
                vmin=vmin_plot,
                vmax=vmax_plot,
                cmap=cmap,
            )
            fig.colorbar(im, ax=ax, shrink=0.8)
            ax.set_title(f"{which.capitalize()} weight matrix – {layer_name}")
            ax.set_xlabel("Input dim")
            ax.set_ylabel("Output dim")
            ax.grid(which="minor", color="w", linestyle="-", linewidth=0.5)
            ax.tick_params(
                which="both",
                bottom=False,
                left=False,
                labelbottom=False,
                labelleft=False,
            )

        parts = ["heat", layer_name, which]
        if epoch is not None:
            parts.append(f"ep{epoch:03d}")

        img_arr = None
        if return_img and fig:
            fig.canvas.draw()
            buf = fig.canvas.tostring_rgb()
            arr = np.frombuffer(buf, dtype=np.uint8)
            reported_w, reported_h = fig.canvas.get_width_height()
            total_pixels = arr.size // 3
            scale = int(round((total_pixels / (reported_w * reported_h)) ** 0.5)) or 1
            img_arr = arr.reshape(reported_h * scale, reported_w * scale, 3)

        if fig:
            self._save_or_show(fig, "_".join(parts), return_img)
        return img_arr

    # ------------------------------------------------------------------
    #  Internal helpers
    # ------------------------------------------------------------------
    def _save_or_show(self, fig, fname: str, return_img: bool):
        should_show_interactive = not self.out_dir and not return_img
        if self.out_dir:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(self.out_dir / f"{fname}.png", dpi=150, bbox_inches="tight")
        if should_show_interactive:
            plt.show()
        plt.close(fig)

    def _iter_named_modules(self):
        if hasattr(self.model, "named_modules"):
            for name, mod in self.model.named_modules():
                if name:
                    yield name, mod
        elif hasattr(self.model, "layers"):
            for idx, layer in enumerate(self.model.layers):
                yield str(idx), layer

    def _locate_layer(self, layer_name: str):
        for n, m in self._iter_named_modules():
            if n == layer_name:
                return m
        raise KeyError(f"Layer '{layer_name}' not found.")

    def set_epoch(self, epoch: int):
        self.current_epoch = int(epoch)

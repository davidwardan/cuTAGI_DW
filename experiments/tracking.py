import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List, Optional


class EmbeddingUpdateTracker:
    """
    Tracks the magnitude of embedding updates for each time series across epochs.
    """

    def __init__(self, num_series: int):
        self.num_series = num_series
        self.current_epoch_deltas = np.zeros(num_series, dtype=np.float32)
        self.history: List[np.ndarray] = []
        self.epochs: List[int] = []
        self.current_epoch = 0

    def update(self, indices: np.ndarray, mu_delta: np.ndarray):
        """
        Accumulates the L2 norm of the updates for the given indices.

        Args:
            indices: Array of time series indices (batch_size,)
            mu_delta: Array of update vectors (batch_size, embedding_size)
        """
        # Calculate L2 norm of updates for each sample in batch
        # mu_delta shape: (B, embedding_size)
        update_norms = np.linalg.norm(mu_delta, axis=1)

        # Filter out -1 indices (padding)
        valid_mask = indices != -1
        valid_indices = indices[valid_mask]
        valid_norms = update_norms[valid_mask]

        np.add.at(self.current_epoch_deltas, valid_indices, valid_norms)

    def step_epoch(self):
        """
        Records the accumulated deltas for the current epoch and resets for the next.
        """
        self.history.append(self.current_epoch_deltas.copy())
        self.epochs.append(self.current_epoch)
        self.current_epoch += 1
        self.current_epoch_deltas.fill(0.0)

    def plot(self, save_dir: str, filename: str = "embedding_updates.png"):
        """
        Plots the history of embedding updates for each series.
        """
        if not self.history:
            print("No embedding update history to plot.")
            return

        # Shape: (num_epochs, num_series)
        history_array = np.stack(self.history, axis=0)

        plt.figure(figsize=(12, 6))

        # Plot each series
        for i in range(self.num_series):
            plt.plot(
                self.epochs,
                history_array[:, i],
                label=f"Series {i}",
                alpha=0.5,
                linewidth=1,
            )

        plt.xlabel("Epoch")
        plt.ylabel("Total Embedding Update (L2 Norm)")
        plt.title("Embedding Updates per Series per Epoch")
        # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small', ncol=2)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, filename), dpi=300)
        plt.close()
        print(f"Saved embedding update plot to {os.path.join(save_dir, filename)}")

    @staticmethod
    def track_embedding_coordinates(
        embeddings,
        config,
        coords_history: list,
    ):
        """
        Snapshot embeddings in 2D using PCA and optionally log to W&B.

        Uses embeddings.as_coordinates(n=2) -> (mu_coords, var_coords).

        - coords_history: list to which (num_embeddings, 2) mu_coords arrays are appended.
        """
        if embeddings is None:
            return
        if not hasattr(embeddings, "as_coordinates"):
            return
        if config.total_embedding_size <= 0:
            return

        # (num_embeddings, 2) for means; we ignore the projected variances for now
        mu_coords, var_coords = embeddings.as_coordinates(n=2)
        coords_history.append(mu_coords.copy())


class ParameterTracker:
    """
    Tracks the mean and variance of specific model parameters over epochs.
    """

    def __init__(self):
        # List of dicts: {"layer_idx": int, "param_type": str, "index": int, "label": str}
        self.tracked_params = []

        # Storage: {label: {"mu": [], "var": []}}
        self.history = {}
        self.epochs = []
        self.current_epoch = 0

    def track_parameter(
        self,
        layer_name: str,
        param_type: str,
        label: str,
        indices: Optional[list] = None,
    ):
        """
        Registers a parameter to track.

        Args:
            layer_name: Name of the layer in state_dict (e.g., "LSTM.0", "Linear.2")
            param_type: "weight" or "bias" (corresponds to mu_w/var_w or mu_b/var_b)
            label: Unique label for plotting
            indices: List of flat indices of the parameter in the array. If None, tracks all.
        """
        if isinstance(indices, int):
            indices = [indices]

        if label in self.history:
            print(f"Warning: Parameter label '{label}' already exists. Overwriting.")

        self.tracked_params.append(
            {
                "layer_name": layer_name,
                "param_type": param_type,
                "indices": indices,
                "label": label,
            }
        )
        # History stores lists of lists: [epoch][index_in_indices]
        self.history[label] = {"mu": [], "var": []}

    def step_epoch(self, net):
        """
        Extracts current mu/var for registered parameters and stores them.
        """
        state_dict = net.state_dict()
        # state_dict keys are like "LSTM.0", "Linear.2", etc.
        # Values are tuples: (mu_w, var_w, mu_b, var_b)

        for param_info in self.tracked_params:
            layer_name = param_info["layer_name"]
            param_type = param_info["param_type"]
            indices = param_info["indices"]
            label = param_info["label"]

            if layer_name not in state_dict:
                print(
                    f"Warning: Layer '{layer_name}' not found in state_dict. Skipping {label}."
                )
                continue

            mu_w, var_w, mu_b, var_b = state_dict[layer_name]

            current_mus = []
            current_vars = []

            if param_type == "weight":
                vals_mu = mu_w
                vals_var = var_w
            elif param_type == "bias":
                vals_mu = mu_b
                vals_var = var_b
            else:
                print(f"Warning: Unknown param_type '{param_type}'. Skipping {label}.")
                continue

            # If indices is None, track all indices
            if indices is None:
                indices = list(range(len(vals_mu)))
                # Update the stored indices so we don't have to regenerate them every time
                # and so plotting knows what indices were tracked.
                param_info["indices"] = indices

            for idx in indices:
                if idx < len(vals_mu):
                    current_mus.append(vals_mu[idx])
                    current_vars.append(vals_var[idx])
                else:
                    print(
                        f"Warning: Index {idx} out of bounds for {layer_name} {param_type}. Skipping."
                    )
                    current_mus.append(np.nan)
                    current_vars.append(np.nan)

            self.history[label]["mu"].append(current_mus)
            self.history[label]["var"].append(current_vars)

        self.epochs.append(self.current_epoch)
        self.current_epoch += 1

    def plot(self, save_dir: str):
        """
        Plots mu and var evolution for each tracked parameter.
        """
        if not self.history:
            print("No parameter history to plot.")
            return

        os.makedirs(save_dir, exist_ok=True)

        for label, data in self.history.items():
            # mus shape: (epochs, num_indices)
            mus = np.array(data["mu"])
            vars = np.array(data["var"])
            stds = np.sqrt(vars)

            # Retrieve indices for legend
            # Find the corresponding tracked_param entry
            indices = []
            for tp in self.tracked_params:
                if tp["label"] == label:
                    indices = tp["indices"]
                    break

            plt.figure(figsize=(10, 8))

            # Plot Mean
            plt.subplot(2, 1, 1)
            for i, idx in enumerate(indices):
                plt.plot(self.epochs, mus[:, i], label=f"Idx {idx}")
                plt.fill_between(
                    self.epochs,
                    mus[:, i] - stds[:, i],
                    mus[:, i] + stds[:, i],
                    alpha=0.1,
                )

            plt.title(f"Parameter Evolution: {label} (Mean)")
            plt.ylabel("Mean Value")
            # plt.legend()
            plt.grid(True, alpha=0.3)

            # Plot Variance
            plt.subplot(2, 1, 2)
            for i, idx in enumerate(indices):
                plt.plot(self.epochs, vars[:, i], label=f"Idx {idx}")

            plt.title(f"Parameter Evolution: {label} (Variance)")
            plt.xlabel("Epoch")
            plt.ylabel("Variance Value")
            # plt.legend()
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            filename = f"param_evolution_{label.replace(' ', '_')}.png"
            plt.savefig(os.path.join(save_dir, filename), dpi=300)
            plt.close()
            print(f"Saved parameter plot to {os.path.join(save_dir, filename)}")


class EpistemicUncertaintyTracker:
    """
    Tracks the epoch-wise distribution of epistemic uncertainty via fixed log-space
    histograms so memory stays bounded regardless of dataset size.
    """

    def __init__(
        self,
        num_bins: int = 80,
        log10_min: float = -12.0,
        log10_max: float = 6.0,
    ):
        if num_bins <= 0:
            raise ValueError("num_bins must be positive.")
        if log10_max <= log10_min:
            raise ValueError("log10_max must be greater than log10_min.")

        self.bin_edges = np.linspace(
            log10_min, log10_max, num_bins + 1, dtype=np.float32
        )
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2.0
        self.current_counts = np.zeros(num_bins, dtype=np.int64)
        self.history: List[np.ndarray] = []
        self.sample_counts: List[int] = []
        self.epochs: List[int] = []
        self._tiny = np.finfo(np.float32).tiny

    def update(self, v_pred: np.ndarray, mask: Optional[np.ndarray] = None):
        """
        Adds a batch of epistemic variances to the current epoch histogram.

        Args:
            v_pred: Array of predictive epistemic variances.
            mask: Optional boolean mask to keep only valid target positions.
        """
        values = np.asarray(v_pred, dtype=np.float32).reshape(-1)
        if values.size == 0:
            return

        if mask is not None:
            mask = np.asarray(mask, dtype=bool).reshape(-1)
            if mask.shape != values.shape:
                raise ValueError("mask and v_pred must have the same flattened shape.")
            values = values[mask]
            if values.size == 0:
                return

        finite_mask = np.isfinite(values)
        if not np.any(finite_mask):
            return

        values = np.maximum(values[finite_mask], self._tiny)
        log_values = np.log10(values)
        counts, _ = np.histogram(log_values, bins=self.bin_edges)
        self.current_counts += counts.astype(np.int64)

    def step_epoch(self, epoch: Optional[int] = None):
        """
        Saves the current histogram and resets the accumulator.
        """
        self.history.append(self.current_counts.copy())
        self.sample_counts.append(int(self.current_counts.sum()))
        self.epochs.append(len(self.epochs) if epoch is None else int(epoch))
        self.current_counts.fill(0)

    def _history_counts(self) -> np.ndarray:
        if not self.history:
            raise ValueError("No epistemic uncertainty history is available.")
        return np.stack(self.history, axis=0).astype(np.float32)

    def _estimate_log_quantiles(self, quantiles: tuple[float, ...]) -> np.ndarray:
        counts = self._history_counts()
        totals = counts.sum(axis=1, keepdims=True)
        frequencies = np.divide(
            counts,
            totals,
            out=np.zeros_like(counts, dtype=np.float32),
            where=totals > 0,
        )
        cdf = np.cumsum(frequencies, axis=1)
        valid_rows = totals.squeeze(-1) > 0

        log_quantiles = np.full(
            (counts.shape[0], len(quantiles)), np.nan, dtype=np.float32
        )
        for idx, quantile in enumerate(quantiles):
            quantile_idx = np.argmax(cdf >= quantile, axis=1)
            log_quantiles[valid_rows, idx] = self.bin_centers[quantile_idx[valid_rows]]

        return log_quantiles

    @staticmethod
    def _relative_to_first_valid(values: np.ndarray) -> np.ndarray:
        relative = np.full(values.shape, np.nan, dtype=np.float64)
        valid_idx = np.flatnonzero(np.isfinite(values) & (values > 0))
        if valid_idx.size == 0:
            return relative

        baseline = values[valid_idx[0]]
        return values / baseline

    def save(
        self,
        save_dir: str,
        filename: str = "train_epistemic_uncertainty_histograms.npz",
    ):
        """
        Saves the histogram counts, bin definitions, and quantile summaries.
        """
        if not self.history:
            print("No epistemic uncertainty history to save.")
            return

        quantile_levels = (0.05, 0.25, 0.5, 0.75, 0.95)
        counts = self._history_counts()
        log_quantiles = self._estimate_log_quantiles(quantile_levels)
        quantiles = np.power(10.0, log_quantiles.astype(np.float64))
        q05, q25, q50, q75, q95 = quantiles.T
        interquartile_spread = q75 / q25
        interquantile_spread = q95 / q05

        os.makedirs(save_dir, exist_ok=True)
        np.savez(
            os.path.join(save_dir, filename),
            epochs=np.asarray(self.epochs, dtype=np.int32),
            log10_bin_edges=self.bin_edges,
            log10_bin_centers=self.bin_centers,
            counts=counts,
            sample_counts=np.asarray(self.sample_counts, dtype=np.int64),
            quantile_levels=np.asarray(quantile_levels, dtype=np.float32),
            log10_quantiles=log_quantiles,
            quantiles=quantiles,
            median_relative_to_start=self._relative_to_first_valid(q50),
            interquartile_spread=interquartile_spread,
            interquartile_spread_relative_to_start=self._relative_to_first_valid(
                interquartile_spread
            ),
            interquantile_spread=interquantile_spread,
            interquantile_spread_relative_to_start=self._relative_to_first_valid(
                interquantile_spread
            ),
            interdecile_spread=interquantile_spread,
            interdecile_spread_relative_to_start=self._relative_to_first_valid(
                interquantile_spread
            ),
        )

    def plot(
        self,
        save_dir: str,
        filename: str = "train_epistemic_uncertainty_quantile_bands.png",
    ):
        """
        Plots epoch-wise epistemic quantile bands and median trajectory.
        """
        if not self.history:
            print("No epistemic uncertainty history to plot.")
            return

        quantile_levels = (0.05, 0.25, 0.5, 0.75, 0.95)
        log_quantiles = self._estimate_log_quantiles(quantile_levels)
        q05, q25, q50, q75, q95 = np.power(10.0, log_quantiles.astype(np.float64)).T
        epochs = np.asarray(self.epochs, dtype=np.int32)
        if epochs.size == 0:
            print("No epochs available for plotting.")
            return
        display_epochs = epochs + 1

        fig, ax = plt.subplots(figsize=(14, 5.2))
        ax.fill_between(
            display_epochs,
            q05,
            q95,
            color="#8ec5ff",
            alpha=0.35,
            label="q05-q95",
        )
        ax.fill_between(
            display_epochs,
            q25,
            q75,
            color="#4d89e7",
            alpha=0.35,
            label="q25-q75",
        )
        ax.plot(
            display_epochs,
            q50,
            color="#1f3b8f",
            linewidth=2.5,
            label="median (q50)",
        )

        ax.set_xlabel("Epoch")
        ax.set_ylabel(r"$v_{pred}$")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")

        tick_step = 1
        tick_start = int(display_epochs[0])
        tick_stop = int(display_epochs[-1])
        xticks = np.arange(tick_start, tick_stop + 1, tick_step, dtype=np.int32)
        if xticks.size == 0 or xticks[0] != tick_start:
            xticks = np.insert(xticks, 0, tick_start)
        if xticks[-1] != tick_stop:
            xticks = np.append(xticks, tick_stop)
        ax.set_xticks(xticks)
        ax.set_xlim(display_epochs[0], display_epochs[-1])

        fig.tight_layout()
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, filename), dpi=300)
        plt.close(fig)
        print(
            f"Saved epistemic uncertainty quantile-band plot to {os.path.join(save_dir, filename)}"
        )

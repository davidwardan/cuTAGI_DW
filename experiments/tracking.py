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

        # Accumulate norms for each series
        # We need to handle potential duplicate indices in a batch if that ever happens,
        # though usually batches are unique series or handled sequentially.
        # Using np.add.at is safe for duplicate indices.

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
        self, layer_name: str, param_type: str, indices: list, label: str
    ):
        """
        Registers a parameter to track.

        Args:
            layer_name: Name of the layer in state_dict (e.g., "LSTM.0", "Linear.2")
            param_type: "weight" or "bias" (corresponds to mu_w/var_w or mu_b/var_b)
            indices: List of flat indices of the parameter in the array
            label: Unique label for plotting
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

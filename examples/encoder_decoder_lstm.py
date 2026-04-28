# Temporary import. It will be removed in the final vserion
import os
import sys

# Add the 'build' directory to sys.path in one line
sys.path.append(
    os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "build"))
)

import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

import pytagi.cuda as cuda
from pytagi import exponential_scheduler
from pytagi.nn import LSTM, Linear, OutputUpdater, Sequential


TRAIN_CSV = "data/hq/ts_weekly_values_final.csv"
SAVE_DIR = "saved_results"


def _trim_trailing_nans(series: np.ndarray) -> np.ndarray:
    """Drop NaN-only tail values from one time series."""
    valid_idx = np.flatnonzero(~np.isnan(series))
    if valid_idx.size == 0:
        return series[:0]
    return series[: valid_idx[-1] + 1]


def load_traffic_windows(csv_path: str, window_len: int, stride: int):
    """Load the traffic CSV and turn every column (one time series per sensor)
    into overlapping windows of length ``window_len``.

    Returns
    -------
    windows : (N, window_len, 1) float32
    first_windows : (num_series, window_len, 1) float32, first valid window
                    for every series with at least one usable window
    """
    df = pd.read_csv(csv_path, skiprows=1, delimiter=",", header=None)
    data = df.values.astype(np.float32)  # (T, num_series)

    windows = []
    first_windows = []
    num_series = data.shape[1]
    for s in range(num_series):
        series = _trim_trailing_nans(data[:, s])
        if series.size < window_len:
            continue

        series_mean = np.nanmean(series)
        series_std = np.nanstd(series)
        if np.isnan(series_mean) or np.isnan(series_std):
            continue
        if series_std < 1e-6:
            series_std = 1.0

        series = (series - series_mean) / series_std
        first_valid_window = None
        for start in range(0, series.size - window_len + 1, stride):
            window = series[start : start + window_len]
            if np.isnan(window).any():
                continue
            windows.append(window)
            if first_valid_window is None:
                first_valid_window = window

        if first_valid_window is not None:
            first_windows.append(first_valid_window)

    if not windows:
        raise ValueError(
            f"No valid windows found in {csv_path!r}; "
            f"window_len={window_len}, stride={stride}."
        )

    windows = np.stack(windows).astype(np.float32)[..., np.newaxis]
    first_windows = np.stack(first_windows).astype(np.float32)[..., np.newaxis]
    return windows, first_windows


def pca_3d(X: np.ndarray) -> np.ndarray:
    """Principal component projection onto 3 dimensions via SVD."""
    Xc = X - X.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:3].T


def repeat_embedding(
    m_z: np.ndarray,
    v_z: np.ndarray,
    window_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Repeat one embedding per sample across all decoder timesteps."""
    m_dec = np.repeat(m_z[:, np.newaxis, :], window_len, axis=1).astype(np.float32)
    v_dec = np.repeat(v_z[:, np.newaxis, :], window_len, axis=1).astype(np.float32)
    return m_dec, v_dec


def aggregate_repeat_deltas(
    delta_mu: np.ndarray,
    delta_var: np.ndarray,
    batch_size: int,
    window_len: int,
    embed_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fold decoder-input deltas from repeated z_t copies back to z."""
    delta_mu = delta_mu.reshape(batch_size, window_len, embed_size).sum(axis=1)
    delta_var = delta_var.reshape(batch_size, window_len, embed_size).sum(axis=1)
    return delta_mu.reshape(-1), delta_var.reshape(-1)


def reconstruct(
    encoder: Sequential,
    decoder: Sequential,
    x: np.ndarray,
    batch_size: int,
    window_len: int,
    embed_size: int,
) -> tuple:
    """Run ``x`` (shape (N, window_len, 1)) through the encoder-decoder and
    return ``(mean, epistemic_var)``, each of shape (N, window_len)."""
    n = x.shape[0]
    pad = (-n) % batch_size
    if pad:
        x_padded = np.concatenate(
            [x, np.zeros((pad, window_len, 1), dtype=np.float32)], axis=0
        )
    else:
        x_padded = x

    mean = np.empty((x_padded.shape[0], window_len), dtype=np.float32)
    var = np.empty((x_padded.shape[0], window_len), dtype=np.float32)
    for start in range(0, x_padded.shape[0], batch_size):
        x_batch = x_padded[start : start + batch_size]

        m_enc, v_enc = encoder(x_batch)
        m_enc = np.asarray(m_enc, dtype=np.float32).reshape(batch_size, embed_size)
        v_enc = np.asarray(v_enc, dtype=np.float32).reshape(batch_size, embed_size)

        m_dec_in, v_dec_in = repeat_embedding(m_enc, v_enc, window_len)
        m_pred, v_pred = decoder(m_dec_in, v_dec_in)
        mean[start : start + batch_size] = np.asarray(m_pred, dtype=np.float32).reshape(
            batch_size, window_len
        )
        var[start : start + batch_size] = np.asarray(v_pred, dtype=np.float32).reshape(
            batch_size, window_len
        )

    return mean[:n], var[:n]


def plot_reconstructions(
    inputs: np.ndarray,
    recons: np.ndarray,
    var_epistemic: np.ndarray,
    sigma_v: float,
    epoch: int,
    save_dir: str,
    std_factor: float = 1.0,
) -> None:
    """Plot ``inputs`` vs ``recons`` with uncertainty bands and save to disk.

    - dark band: epistemic only (model uncertainty from the decoder).
    - light band: epistemic + aleatoric (sigma_v^2 added on top).
    """
    n = inputs.shape[0]
    n_cols = 5
    n_rows = int(np.ceil(n / n_cols))
    std_ep = np.sqrt(np.maximum(var_epistemic, 0.0))
    std_tot = np.sqrt(np.maximum(var_epistemic + sigma_v**2, 0.0))
    t = np.arange(inputs.shape[1])

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3 * n_cols, 2 * n_rows), sharex=True
    )
    axes = np.atleast_2d(axes)
    for i in range(n_rows * n_cols):
        ax = axes[i // n_cols, i % n_cols]
        if i < n:
            ax.fill_between(
                t,
                recons[i] - std_factor * std_tot[i],
                recons[i] + std_factor * std_tot[i],
                color="red",
                alpha=0.15,
                label=f"$\\pm{std_factor:g}\\sigma$ (ep.+al.)",
            )
            ax.fill_between(
                t,
                recons[i] - std_factor * std_ep[i],
                recons[i] + std_factor * std_ep[i],
                color="red",
                alpha=0.35,
                label=f"$\\pm{std_factor:g}\\sigma$ (epistemic)",
            )
            ax.plot(t, inputs[i], color="black", lw=1.5, label="input")
            ax.plot(t, recons[i], color="red", lw=1.5, label="recon")
            ax.set_title(f"#{i}", fontsize=9)
            if i == 0:
                ax.legend(fontsize=7, loc="upper right")
        else:
            ax.axis("off")
    fig.suptitle(
        f"Epoch {epoch}: input vs reconstruction (sigma_v={sigma_v:.3f})",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = os.path.join(save_dir, f"reconstruction_epoch_{epoch:03d}.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main(
    num_epochs: int = 30,
    batch_size: int = 64,
    sigma_v: float = 1.0,
    window_len: int = 52,
    embed_size: int = 20,
    hidden_size: int = 50,
    stride: int = 1,
    cuda_index: int = 0,
    train_csv: str = TRAIN_CSV,
    output_prefix: str = "hq",
):
    """Train an LSTM encoder-decoder that compresses a window of a traffic
    time series into a low-dimensional embedding and reconstructs it."""

    os.makedirs(SAVE_DIR, exist_ok=True)

    # --------------------------------------------------------------------- #
    # Data
    x_all, first_windows = load_traffic_windows(
        train_csv, window_len=window_len, stride=stride
    )
    num_samples = x_all.shape[0]
    num_series = first_windows.shape[0]
    print(
        f"Loaded {num_samples} windows of length {window_len} "
        f"from {num_series} series with at least one valid window."
    )

    # --------------------------------------------------------------------- #
    # Network: encoder (many-to-one) + decoder (one-to-many)
    encoder = Sequential(
        LSTM(1, hidden_size, last_timestep=False, seq_len=window_len),
        LSTM(
            hidden_size,
            embed_size,
            last_timestep=True,
            seq_len=window_len,
        ),
    )

    # Decoder: consumes z repeated at every timestep, then maps each decoder
    # hidden state to the reconstructed scalar x_t.
    decoder = Sequential(
        LSTM(embed_size, hidden_size, last_timestep=False, seq_len=window_len),
        Linear(hidden_size, 1),
    )

    device = f"cuda:{cuda_index}" if cuda.is_available() else "cpu"
    encoder.to_device(device)
    decoder.to_device(device)
    out_updater = OutputUpdater(decoder.device)

    var_y = np.full((batch_size * window_len,), sigma_v**2, dtype=np.float32)

    # --------------------------------------------------------------------- #
    # Training
    pbar = tqdm(range(num_epochs), desc="Training")
    for epoch in pbar:
        perm = np.random.permutation(num_samples)
        x_shuffled = x_all[perm]

        sigma_v = exponential_scheduler(
            curr_v=sigma_v, min_v=0.2, decaying_factor=0.8, curr_iter=epoch
        )
        var_y = np.full((batch_size * window_len,), sigma_v**2, dtype=np.float32)

        mses = []
        num_batches = num_samples // batch_size
        decoder_input_slots = batch_size * window_len * embed_size
        embed_slots = batch_size * embed_size
        for b in range(num_batches):
            x_batch = x_shuffled[b * batch_size : (b + 1) * batch_size]
            x_flat = x_batch.reshape(-1).astype(np.float32)

            # Encode: (batch, window_len, 1) -> (batch, embed_size)
            m_enc, v_enc = encoder(x_batch)
            m_enc = np.asarray(m_enc, dtype=np.float32).reshape(batch_size, embed_size)
            v_enc = np.asarray(v_enc, dtype=np.float32).reshape(batch_size, embed_size)

            # Repeat z T times: (batch, embed_size) -> (batch, window_len, embed_size)
            m_dec_in, v_dec_in = repeat_embedding(m_enc, v_enc, window_len)

            # Decode: reconstruct the window
            m_pred, _ = decoder(m_dec_in, v_dec_in)

            # Reconstruction loss at the output
            out_updater.update(
                output_states=decoder.output_z_buffer,
                mu_obs=x_flat,
                var_obs=var_y,
                delta_states=decoder.input_delta_z_buffer,
            )

            decoder.backward()
            decoder.step()

            # Propagate deltas: decoder -> repeated z -> encoder
            delta_mu_dec = np.asarray(
                decoder.output_delta_z_buffer.delta_mu, dtype=np.float32
            )[:decoder_input_slots]
            delta_var_dec = np.asarray(
                decoder.output_delta_z_buffer.delta_var, dtype=np.float32
            )[:decoder_input_slots]
            delta_mu_agg, delta_var_agg = aggregate_repeat_deltas(
                delta_mu_dec,
                delta_var_dec,
                batch_size=batch_size,
                window_len=window_len,
                embed_size=embed_size,
            )
            delta_mu_agg = delta_mu_agg[:embed_slots]
            delta_var_agg = delta_var_agg[:embed_slots]

            encoder.set_delta_z(delta_mu_agg, delta_var_agg)
            encoder.backward()
            encoder.step()

            mses.append(float(np.nanmean((m_pred - x_flat) ** 2)))

        # Flush any carried-over LSTM state at epoch boundary.
        encoder.reset_lstm_states()
        decoder.reset_lstm_states()

        # Sample 10 random windows and save their reconstructions with
        # uncertainty bands (epistemic vs epistemic + aleatoric).
        sample_idx = np.random.choice(num_samples, size=10, replace=False)
        sample_x = x_all[sample_idx]
        sample_recon, sample_var = reconstruct(
            encoder,
            decoder,
            sample_x,
            batch_size=batch_size,
            window_len=window_len,
            embed_size=embed_size,
        )
        plot_reconstructions(
            inputs=sample_x[:, :, 0],
            recons=sample_recon,
            var_epistemic=sample_var,
            sigma_v=sigma_v,
            epoch=epoch + 1,
            save_dir=SAVE_DIR,
        )
        encoder.reset_lstm_states()
        decoder.reset_lstm_states()

        pbar.set_description(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"mse={sum(mses) / len(mses):.4f} | sigma_v={sigma_v:.3f}",
            refresh=True,
        )
    pbar.close()

    # --------------------------------------------------------------------- #
    # Embedding extraction: one embedding per sensor (first window)
    encoder.eval()

    emb_mean = np.zeros((num_series, embed_size), dtype=np.float32)
    emb_var = np.zeros((num_series, embed_size), dtype=np.float32)
    for start in range(0, num_series, batch_size):
        end = min(start + batch_size, num_series)
        batch = first_windows[start:end]
        if batch.shape[0] < batch_size:
            pad = np.zeros(
                (batch_size - batch.shape[0], window_len, 1), dtype=np.float32
            )
            batch_padded = np.concatenate([batch, pad], axis=0)
        else:
            batch_padded = batch

        # reset between batches so each window is encoded from a fresh state
        encoder.reset_lstm_states()
        mu_out, var_out = encoder(batch_padded)
        mu_out = np.asarray(mu_out, dtype=np.float32).reshape(batch_size, embed_size)
        var_out = np.asarray(var_out, dtype=np.float32).reshape(batch_size, embed_size)
        emb_mean[start:end] = mu_out[: end - start]
        emb_var[start:end] = var_out[: end - start]

    mean_path = os.path.join(SAVE_DIR, f"{output_prefix}_embeddings_mean.csv")
    var_path = os.path.join(SAVE_DIR, f"{output_prefix}_embeddings_var.csv")
    np.savetxt(mean_path, emb_mean, delimiter=",")
    np.savetxt(var_path, emb_var, delimiter=",")
    print(f"Saved mean embeddings -> {mean_path}  (shape {emb_mean.shape})")
    print(f"Saved var  embeddings -> {var_path}  (shape {emb_var.shape})")

    # --------------------------------------------------------------------- #
    # # 3D PCA visualisation (mean embeddings)
    # if not np.isfinite(emb_mean).all():
    #     print(
    #         "Skipping PCA figure: mean embeddings contain non-finite values."
    #     )
    #     return
    # embeds_3d = pca_3d(emb_mean)

    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111, projection="3d")
    # sc = ax.scatter(
    #     embeds_3d[:, 0],
    #     embeds_3d[:, 1],
    #     embeds_3d[:, 2],
    #     c=np.arange(num_series),
    #     cmap="viridis",
    #     s=12,
    #     alpha=0.8,
    # )
    # ax.set_xlabel("PC1")
    # ax.set_ylabel("PC2")
    # ax.set_zlabel("PC3")
    # ax.set_title(f"PCA of LSTM embeddings ({output_prefix})")
    # fig.colorbar(sc, ax=ax, shrink=0.6, label="sensor index")

    # fig_path = os.path.join(SAVE_DIR, f"{output_prefix}_embedding_pca.png")
    # plt.savefig(fig_path, bbox_inches="tight", dpi=150)
    # plt.close(fig)
    # print(f"Saved 3D PCA figure -> {fig_path}")


if __name__ == "__main__":
    fire.Fire(main)

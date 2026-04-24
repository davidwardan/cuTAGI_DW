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
from pytagi.nn import LSTM, OutputUpdater, Sequential


TRAIN_CSV = "data/traffic/traffic_2008_01_14_train.csv"
SAVE_DIR = "saved_results"


def load_traffic_windows(csv_path: str, window_len: int, stride: int):
    """Load the traffic CSV and turn every column (one time series per sensor)
    into overlapping windows of length ``window_len``.

    Returns
    -------
    windows : (N, window_len, 1) float32
    raw_std : (T, num_series) float32, standardised series (used later to
              embed one representative window per series)
    """
    df = pd.read_csv(csv_path, skiprows=1, delimiter=",", header=None)
    data = df.values.astype(np.float32)  # (T, num_series)

    series_mean = data.mean(axis=0)
    series_std = data.std(axis=0)
    series_std[series_std < 1e-6] = 1.0
    raw_std = (data - series_mean) / series_std

    T, num_series = raw_std.shape
    windows = []
    for s in range(num_series):
        series = raw_std[:, s]
        for start in range(0, T - window_len + 1, stride):
            windows.append(series[start : start + window_len])

    windows = np.stack(windows).astype(np.float32)[..., np.newaxis]
    return windows, raw_std


def pca_3d(X: np.ndarray) -> np.ndarray:
    """Principal component projection onto 3 dimensions via SVD."""
    Xc = X - X.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:3].T


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
        m_enc = np.asarray(m_enc, dtype=np.float32).reshape(
            batch_size, embed_size
        )
        v_enc = np.asarray(v_enc, dtype=np.float32).reshape(
            batch_size, embed_size
        )

        m_dec_in = np.zeros(
            (batch_size, window_len, embed_size), dtype=np.float32
        )
        v_dec_in = np.zeros_like(m_dec_in)
        m_dec_in[:, 0, :] = m_enc
        v_dec_in[:, 0, :] = v_enc

        m_pred, v_pred = decoder(m_dec_in, v_dec_in)
        mean[start : start + batch_size] = np.asarray(
            m_pred, dtype=np.float32
        ).reshape(batch_size, window_len)
        var[start : start + batch_size] = np.asarray(
            v_pred, dtype=np.float32
        ).reshape(batch_size, window_len)

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
    num_epochs: int = 20,
    batch_size: int = 128,
    sigma_v: float = 1.0,
    window_len: int = 24,
    embed_size: int = 10,
    hidden_size: int = 64,
    stride: int = 6,
    cuda_index: int = 0,
):
    """Train an LSTM encoder-decoder that compresses a window of a traffic
    time series into a low-dimensional embedding and reconstructs it."""

    os.makedirs(SAVE_DIR, exist_ok=True)

    # --------------------------------------------------------------------- #
    # Data
    x_all, raw_std = load_traffic_windows(
        TRAIN_CSV, window_len=window_len, stride=stride
    )
    num_samples = x_all.shape[0]
    num_series = raw_std.shape[1]
    print(
        f"Loaded {num_samples} windows of length {window_len} "
        f"from {num_series} series."
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

    # The decoder's first LSTM consumes a repeated copy of the embedding at every
    # output timestep (we build that tensor in the training loop).
    decoder = Sequential(
        LSTM(embed_size, hidden_size, last_timestep=False, seq_len=window_len),
        LSTM(hidden_size, 1, last_timestep=False, seq_len=window_len),
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
            curr_v=sigma_v, min_v=0.5, decaying_factor=0.8, curr_iter=epoch
        )
        var_y = np.full(
            (batch_size * window_len,), sigma_v**2, dtype=np.float32
        )

        mses = []
        num_batches = num_samples // batch_size
        embed_slots = batch_size * window_len * embed_size
        for b in range(num_batches):
            x_batch = x_shuffled[b * batch_size : (b + 1) * batch_size]
            x_flat = x_batch.reshape(-1).astype(np.float32)

            # Encode: (batch, window_len, 1) -> (batch, embed_size)
            m_enc, v_enc = encoder(x_batch)
            m_enc = np.asarray(m_enc, dtype=np.float32).reshape(
                batch_size, embed_size
            )
            v_enc = np.asarray(v_enc, dtype=np.float32).reshape(
                batch_size, embed_size
            )

            # Inject the embedding only at t=0 and feed zeros afterwards; the
            # decoder's LSTM state carries the information across the sequence.
            # This avoids compounding 24 shared-latent deltas (which, when
            # summed, tends to drive the encoder's posterior variance negative
            # and produce NaNs).
            m_dec_in = np.zeros(
                (batch_size, window_len, embed_size), dtype=np.float32
            )
            v_dec_in = np.zeros_like(m_dec_in)
            m_dec_in[:, 0, :] = m_enc
            v_dec_in[:, 0, :] = v_enc

            # Decode: reconstruct the input window
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

            # decoder.output_delta_z_buffer holds deltas for every input
            # position; only t=0 carried the real embedding, so the encoder's
            # output delta is just the first timestep slice.
            d_mu = np.asarray(
                decoder.output_delta_z_buffer.delta_mu, dtype=np.float32
            )[:embed_slots].reshape(batch_size, window_len, embed_size)
            d_var = np.asarray(
                decoder.output_delta_z_buffer.delta_var, dtype=np.float32
            )[:embed_slots].reshape(batch_size, window_len, embed_size)
            delta_mu_agg = d_mu[:, 0, :].reshape(-1)
            delta_var_agg = d_var[:, 0, :].reshape(-1)

            encoder.set_delta_z(delta_mu_agg, delta_var_agg)
            encoder.backward()
            encoder.step()

            mses.append(float(np.mean((m_pred - x_flat) ** 2)))

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
    first_windows = raw_std[:window_len, :].T[:, :, np.newaxis].astype(
        np.float32
    )  # (num_series, window_len, 1)

    embeddings = np.zeros((num_series, embed_size), dtype=np.float32)
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
        mu_out, _ = encoder(batch_padded)
        mu_out = np.asarray(mu_out).reshape(batch_size, embed_size)
        embeddings[start:end] = mu_out[: end - start]

    emb_path = os.path.join(SAVE_DIR, "traffic_embeddings.csv")
    np.savetxt(emb_path, embeddings, delimiter=",")
    print(f"Saved embeddings -> {emb_path}  (shape {embeddings.shape})")

    # --------------------------------------------------------------------- #
    # 3D PCA visualisation
    embeds_3d = pca_3d(embeddings)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        embeds_3d[:, 0],
        embeds_3d[:, 1],
        embeds_3d[:, 2],
        c=np.arange(num_series),
        cmap="viridis",
        s=12,
        alpha=0.8,
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("PCA of LSTM embeddings (traffic sensors)")
    fig.colorbar(sc, ax=ax, shrink=0.6, label="sensor index")

    fig_path = os.path.join(SAVE_DIR, "traffic_embedding_pca.png")
    plt.savefig(fig_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved 3D PCA figure -> {fig_path}")


if __name__ == "__main__":
    fire.Fire(main)

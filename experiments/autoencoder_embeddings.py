import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
import copy
from typing import Tuple

# ==========================================
# 1. HYPERPARAMETERS & CONFIGURATION
# ==========================================
CONFIG = {
    # File Paths
    "train_csv": "data/hq/train100/split_train_values.csv",
    "val_csv": "data/hq/split_val_values.csv",
    "output_plot": "out/embeddings_pca_attn.svg",
    "output_embedding_file": "out/autoencoder_embeddings_attn.npy",
    # Model Params
    "seq_len": 52,  # Window size
    "embedding_dim": 10,  # Latent vector size
    "hidden_dim": 40,  # Internal LSTM size
    # Training Params
    "batch_size": 128,
    "learning_rate": 1e-3,
    "epochs": 100,
    "patience": 10,
    "teacher_forcing_start": 0.5,
    "generate_dummy_data": False,
}

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# 2. DATA PREPARATION UTILS
# ==========================================


def create_dummy_csv(filename_train, filename_val):
    """Generates dummy files if they don't exist."""
    os.makedirs(os.path.dirname(filename_train), exist_ok=True)
    os.makedirs(os.path.dirname(filename_val), exist_ok=True)

    def generate(cols):
        data = {}
        for i in range(cols):
            length = np.random.randint(60, 100)
            if i % 2 == 0:
                ts = np.sin(np.linspace(0, 20, length))
            else:
                ts = np.linspace(0, 1, length) + np.random.normal(0, 0.1, length)
            ts = np.pad(ts, (0, 100 - length), constant_values=np.nan)
            data[f"series_{i}"] = ts
        return pd.DataFrame(data)

    generate(20).to_csv(filename_train, index=False)
    generate(10).to_csv(filename_val, index=False)
    print("Dummy train/val files created.")


def _trim_trailing_nans(x: np.ndarray) -> np.ndarray:
    """Trim padded trailing NaNs and always return a float32 numpy array."""
    arr = np.asarray(x, dtype=np.float32)
    if arr.size == 0:
        return arr

    valid = ~np.isnan(arr)
    if not np.any(valid):
        return np.empty(0, dtype=np.float32)

    last = np.where(valid)[0][-1]
    return arr[: last + 1]


def prepare_data(df, seq_len):
    """
    Reads columns, removes NaNs, STANDARDIZES (Mean/Std), and windows data.
    """
    all_windows = []
    window_ownership = []

    numeric_df = df.select_dtypes(include=[np.number])

    for col in numeric_df.columns:
        series_values = numeric_df[col].to_numpy(dtype=np.float32, copy=True)
        raw_values = _trim_trailing_nans(series_values)

        if raw_values.size < seq_len:
            continue

        finite_values = raw_values[np.isfinite(raw_values)]
        if finite_values.size < seq_len:
            continue

        mean = finite_values.mean()
        std = finite_values.std()
        if std < 1e-6:
            std = 1.0

        scaled_series = ((raw_values - mean) / std).astype(np.float32)

        for i in range(len(scaled_series) - seq_len + 1):
            window = scaled_series[i : i + seq_len]
            if np.isnan(window).any():
                continue
            all_windows.append(window.reshape(seq_len, 1))
            window_ownership.append(col)

    return np.array(all_windows), window_ownership


# ==========================================
# 3. PYTORCH MODEL (ATTENTION + RECURRENT + TANH)
# ==========================================


class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionPooling, self).__init__()
        # Weights to compute a score for each timestep
        self.attention_weights = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_outputs):
        # lstm_outputs: (Batch, Seq_Len, Hidden_Dim)

        # 1. Score each timestep
        scores = self.attention_weights(lstm_outputs)  # (Batch, Seq_Len, 1)

        # 2. Softmax to get probabilities
        weights = torch.softmax(scores, dim=1)

        # 3. Weighted Sum
        # (Batch, Seq_Len, Hidden) * (Batch, Seq_Len, 1) -> Sum dim 1
        context_vector = torch.sum(lstm_outputs * weights, dim=1)  # (Batch, Hidden)

        return context_vector, weights


class LSTMAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim, hidden_dim):
        super(LSTMAutoencoder, self).__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # --- ENCODER ---
        self.encoder_lstm = nn.LSTM(
            input_size=n_features, hidden_size=hidden_dim, batch_first=True
        )

        # Attention Mechanism
        self.attention = AttentionPooling(hidden_dim)

        # Projection to Embedding
        self.encoder_fc = nn.Linear(hidden_dim, embedding_dim)
        self.bn_embedding = nn.BatchNorm1d(embedding_dim)

        # --- DECODER SETUP ---
        self.decoder_fc_hidden = nn.Linear(embedding_dim, hidden_dim)
        self.decoder_fc_cell = nn.Linear(embedding_dim, hidden_dim)
        self.decoder_cell = nn.LSTMCell(input_size=n_features, hidden_size=hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, n_features)

    def forward(self, x, teacher_forcing_ratio=0.0):
        batch_size = x.shape[0]

        # 1. ENCODE
        # encoder_out: (Batch, Seq_Len, Hidden)
        encoder_out, _ = self.encoder_lstm(x)

        # Apply Attention Pooling
        # Collapses Seq_Len dimension by weighted sum
        context_vector, attn_weights = self.attention(encoder_out)

        # Project -> BN -> Tanh (Force range [-1, 1])
        raw_embedding = self.encoder_fc(context_vector)
        embedding = torch.tanh(self.bn_embedding(raw_embedding))

        # 2. DECODE INITIALIZATION
        dec_hidden = self.decoder_fc_hidden(embedding)
        dec_cell = self.decoder_fc_cell(embedding)

        current_input = torch.zeros((batch_size, self.n_features), device=x.device)
        outputs = []

        # 3. DECODE LOOP
        for t in range(self.seq_len):
            dec_hidden, dec_cell = self.decoder_cell(
                current_input, (dec_hidden, dec_cell)
            )

            reconstruction_t = self.output_layer(dec_hidden)
            outputs.append(reconstruction_t.unsqueeze(1))

            if self.training and np.random.random() < teacher_forcing_ratio:
                current_input = x[:, t, :]
            else:
                current_input = reconstruction_t

        reconstructed_seq = torch.cat(outputs, dim=1)

        return reconstructed_seq, embedding, attn_weights


# ==========================================
# 4. MAIN EXECUTION FLOW
# ==========================================

if __name__ == "__main__":

    # --- Step 0: Setup ---
    if CONFIG["generate_dummy_data"]:
        create_dummy_csv(CONFIG["train_csv"], CONFIG["val_csv"])

    # --- Step 1: Load Data ---
    print("Loading Data...")

    if not os.path.exists(CONFIG["train_csv"]):
        print(f"File {CONFIG['train_csv']} not found. Generating dummy data...")
        create_dummy_csv(CONFIG["train_csv"], CONFIG["val_csv"])

    df_train = pd.read_csv(CONFIG["train_csv"])
    X_train_np, train_names = prepare_data(df_train, CONFIG["seq_len"])

    if not os.path.exists(CONFIG["val_csv"]):
        create_dummy_csv(CONFIG["train_csv"], CONFIG["val_csv"])

    df_val = pd.read_csv(CONFIG["val_csv"])
    X_val_np, _ = prepare_data(df_val, CONFIG["seq_len"])

    print(f"Train Windows: {len(X_train_np)} | Val Windows: {len(X_val_np)}")

    if len(X_train_np) == 0:
        raise ValueError("No training windows generated. Check data format.")

    train_tensor = torch.tensor(X_train_np, dtype=torch.float32).to(device)
    val_tensor = torch.tensor(X_val_np, dtype=torch.float32).to(device)

    # Use drop_last=True for BatchNorm safety during training
    train_loader = DataLoader(
        TensorDataset(train_tensor, train_tensor),
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_tensor, val_tensor),
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        drop_last=False,
    )

    # --- Step 2: Initialize Model ---
    model = LSTMAutoencoder(
        seq_len=CONFIG["seq_len"],
        n_features=1,
        embedding_dim=CONFIG["embedding_dim"],
        hidden_dim=CONFIG["hidden_dim"],
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])

    # --- Step 3: Training Loop ---
    print("\nStarting Training (Attention + Recurrent + Tanh)...")

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    for epoch in range(CONFIG["epochs"]):
        # Decay teacher forcing from 0.5 -> 0.0
        tf_ratio = max(
            0.0, CONFIG["teacher_forcing_start"] * (1 - epoch / CONFIG["epochs"])
        )

        # A. TRAINING
        model.train()
        train_loss = 0
        for batch_in, batch_target in train_loader:
            optimizer.zero_grad()
            reconstruction, _, _ = model(batch_in, teacher_forcing_ratio=tf_ratio)
            loss = criterion(reconstruction, batch_target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # B. VALIDATION
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_in, batch_target in val_loader:
                reconstruction, _, _ = model(batch_in, teacher_forcing_ratio=0.0)
                loss = criterion(reconstruction, batch_target)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(
            f"Epoch [{epoch+1}/{CONFIG['epochs']}] "
            f"TF: {tf_ratio:.2f} | "
            f"Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= CONFIG["patience"]:
                print(f"Early stopping triggered at epoch {epoch+1}.")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("\nRestored best model weights.")

    # --- Step 4: Extract Embeddings ---
    print("Extracting Embeddings...")
    model.eval()

    inference_loader = DataLoader(
        TensorDataset(train_tensor, train_tensor),
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        drop_last=False,
    )

    embedding_list = []
    weight_list = []

    with torch.no_grad():
        for batch_in, _ in inference_loader:
            _, batch_embeddings, batch_attn_weights = model(
                batch_in, teacher_forcing_ratio=0.0
            )
            embedding_list.append(batch_embeddings.cpu().numpy())
            weight_list.append(batch_attn_weights.cpu().numpy())

    all_embeddings = np.concatenate(embedding_list, axis=0)
    all_weights = np.concatenate(weight_list, axis=0).squeeze(-1)  # (N, Seq_Len)

    print(
        f"Embedding Range Check: Min={all_embeddings.min():.4f}, Max={all_embeddings.max():.4f}"
    )
    print(
        f"Attention Weights Check: Min={all_weights.min():.6f}, Max={all_weights.max():.6f}, Mean={all_weights.mean():.6f}"
    )

    # --- Step 5: Visualization of Attention Maps ---
    print("Generating Attention Map Visualization...")
    # Visualize the first 10 samples
    num_samples_to_plot = 10
    fig, axes = plt.subplots(
        num_samples_to_plot, 2, figsize=(15, 3 * num_samples_to_plot)
    )

    # Ensure axes is always 2D array even if num_samples_to_plot=1
    if num_samples_to_plot == 1:
        axes = np.expand_dims(axes, 0)

    indices_to_plot = np.random.choice(
        len(train_tensor),
        size=min(num_samples_to_plot, len(train_tensor)),
        replace=False,
    )
    indices_to_plot.sort()  # Sort for consistent order if needed, or just keep random

    for i, idx in enumerate(indices_to_plot):
        # Original Series
        original_series = train_tensor[idx].cpu().numpy().flatten()
        weights = all_weights[idx]

        # Plot Original
        ax_ts = axes[i, 0]
        ax_ts.plot(original_series, label="Time Series", color="blue")
        ax_ts.set_title(f"Sample {idx}: Time Series")
        ax_ts.grid(True, alpha=0.3)

        # Plot Attention
        ax_attn = axes[i, 1]
        ax_attn.plot(weights, label="Attention Weights", color="orange")
        ax_attn.fill_between(range(len(weights)), weights, alpha=0.3, color="orange")
        ax_attn.set_title(f"Sample {idx}: Attention Weights")
        # ax_attn.set_ylim(0, 1.0) # REMOVED: efficient attention might be small values if distributed
        # Auto-scale but ensure 0 is at bottom
        ax_attn.set_ylim(bottom=0)
        ax_attn.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("out/attention_maps_sample.svg")
    print("Attention maps saved to out/attention_maps_sample.svg")

    # --- Step 6: Aggregation ---
    res_df = pd.DataFrame(all_embeddings)
    res_df["series_name"] = train_names
    final_embeddings_df = res_df.groupby("series_name").mean()

    # --- Step 7: PCA Visualization ---
    if len(final_embeddings_df) > 1:
        pca = PCA(n_components=2)
        components = pca.fit_transform(final_embeddings_df.values)

        plt.figure(figsize=(10, 8))
        plt.scatter(components[:, 0], components[:, 1], c="blue", alpha=0.7)

        print("\n=== LEGEND: Index to Series Name (First 20) ===")
        for i, series_name in enumerate(final_embeddings_df.index):
            if i < 20:
                print(f"Index {i}: {series_name}")
            plt.annotate(
                str(i),
                (components[i, 0], components[i, 1]),
                fontsize=9,
                fontweight="bold",
            )

        plt.title(
            f"Attention Embeddings (Tanh)\nLen: {CONFIG['seq_len']}, Dim: {CONFIG['embedding_dim']}"
        )
        plt.xlabel("PC 1")
        plt.ylabel("PC 2")
        plt.grid(True, alpha=0.3)
        plt.savefig(CONFIG["output_plot"])
        print(f"\nPlot saved to {CONFIG['output_plot']}")
    else:
        print("Not enough series to perform PCA.")

    # --- Step 8: Save ---
    print(f"Saving embeddings to {CONFIG['output_embedding_file']}...")
    np.save(CONFIG["output_embedding_file"], final_embeddings_df.values)
    print("Embeddings saved.")
    print("\nDone!")

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
    "output_plot": "embeddings_pca.png",
    # Model Params
    "seq_len": 52,  # Window size
    "embedding_dim": 15,  # Latent vector size
    "hidden_dim": 40,  # Internal LSTM size
    # Training Params
    "batch_size": 128,
    "learning_rate": 1e-3,
    "epochs": 100,  # High max epochs, Early Stopping will cut it short
    "patience": 10,  # Stop if Val Loss doesn't improve for 5 epochs
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

    generate(10).to_csv(filename_train, index=False)
    generate(5).to_csv(filename_val, index=False)
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

    # SAFETY: Only process numeric columns
    numeric_df = df.select_dtypes(include=[np.number])

    for col in numeric_df.columns:
        # 1. Remove trailing NaNs but preserve interior NaNs so we can skip windows later
        series_values = numeric_df[col].to_numpy(dtype=np.float32, copy=True)
        raw_values = _trim_trailing_nans(series_values)

        if raw_values.size < seq_len:
            continue

        finite_values = raw_values[np.isfinite(raw_values)]
        if finite_values.size < seq_len:
            continue  # Not enough real samples to form a clean window

        mean = finite_values.mean()
        std = finite_values.std()
        if std < 1e-6:
            std = 1.0

        scaled_series = ((raw_values - mean) / std).astype(np.float32)

        # 3. Sliding Window (skip any window containing NaNs)
        for i in range(len(scaled_series) - seq_len + 1):
            window = scaled_series[i : i + seq_len]
            if np.isnan(window).any():
                continue
            all_windows.append(window.reshape(seq_len, 1))
            window_ownership.append(col)

    return np.array(all_windows), window_ownership


# ==========================================
# 3. PYTORCH MODEL (LSTM AUTOENCODER)
# ==========================================


class LSTMAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim, hidden_dim):
        super(LSTMAutoencoder, self).__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Encoder
        self.encoder_lstm = nn.LSTM(
            input_size=n_features, hidden_size=hidden_dim, batch_first=True
        )
        self.encoder_fc = nn.Linear(hidden_dim, embedding_dim)

        # Decoder
        self.decoder_fc = nn.Linear(embedding_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True
        )
        self.output_layer = nn.Linear(hidden_dim, n_features)

    def forward(self, x):
        # --- ENCODE ---
        _, (hidden, _) = self.encoder_lstm(x)
        hidden = hidden.squeeze(0)
        embedding = self.encoder_fc(hidden)

        # --- DECODE ---
        decoder_input = self.decoder_fc(embedding)
        decoder_input = decoder_input.unsqueeze(1).repeat(1, self.seq_len, 1)
        decoded_out, _ = self.decoder_lstm(decoder_input)
        reconstruction = self.output_layer(decoded_out)

        return reconstruction, embedding


# ==========================================
# 4. MAIN EXECUTION FLOW
# ==========================================

if __name__ == "__main__":

    # --- Step 0: Setup ---
    if CONFIG["generate_dummy_data"]:
        create_dummy_csv(CONFIG["train_csv"], CONFIG["val_csv"])

    # --- Step 1: Load and Process Data ---
    print("Loading Data...")

    # A. Load Training Data
    if not os.path.exists(CONFIG["train_csv"]):
        raise FileNotFoundError(f"Missing: {CONFIG['train_csv']}")
    df_train = pd.read_csv(CONFIG["train_csv"])
    X_train_np, train_names = prepare_data(df_train, CONFIG["seq_len"])

    # B. Load Validation Data
    if not os.path.exists(CONFIG["val_csv"]):
        raise FileNotFoundError(f"Missing: {CONFIG['val_csv']}")
    df_val = pd.read_csv(CONFIG["val_csv"])
    X_val_np, _ = prepare_data(df_val, CONFIG["seq_len"])

    print(f"Train Windows: {len(X_train_np)} | Val Windows: {len(X_val_np)}")

    if len(X_train_np) == 0:
        raise ValueError("No training windows generated.")

    # Convert to Tensors
    train_tensor = torch.tensor(X_train_np, dtype=torch.float32).to(device)
    val_tensor = torch.tensor(X_val_np, dtype=torch.float32).to(device)

    # Create DataLoaders
    train_loader = DataLoader(
        TensorDataset(train_tensor, train_tensor),
        batch_size=CONFIG["batch_size"],
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_tensor, val_tensor),
        batch_size=CONFIG["batch_size"],
        shuffle=False,
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

    # --- Step 3: Training Loop with Early Stopping ---
    print("\nStarting Training...")

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    for epoch in range(CONFIG["epochs"]):
        # A. TRAINING PHASE
        model.train()
        train_loss = 0
        for batch_in, batch_target in train_loader:
            optimizer.zero_grad()
            reconstruction, _ = model(batch_in)
            loss = criterion(reconstruction, batch_target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # B. VALIDATION PHASE
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_in, batch_target in val_loader:
                reconstruction, _ = model(batch_in)
                loss = criterion(reconstruction, batch_target)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(
            f"Epoch [{epoch+1}/{CONFIG['epochs']}] "
            f"Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}"
        )

        # C. EARLY STOPPING CHECK
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= CONFIG["patience"]:
                print(
                    f"Early stopping triggered! No improvement for {CONFIG['patience']} epochs."
                )
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("\nRestored best model weights.")

    # --- Step 4: Extract Embeddings (From Training Set) ---
    print("Extracting Embeddings...")
    model.eval()

    inference_loader = DataLoader(
        TensorDataset(train_tensor, train_tensor),
        batch_size=CONFIG["batch_size"],
        shuffle=False,
    )

    embedding_list = []

    with torch.no_grad():
        for batch_in, _ in inference_loader:
            _, batch_embeddings = model(batch_in)
            embedding_list.append(batch_embeddings.cpu().numpy())

    all_embeddings = np.concatenate(embedding_list, axis=0)

    # --- Step 5: Aggregation ---
    res_df = pd.DataFrame(all_embeddings)
    res_df["series_name"] = train_names
    final_embeddings_df = res_df.groupby("series_name").mean()

    # --- Step 6: PCA Visualization (LABELED WITH INDEX) ---
    if len(final_embeddings_df) > 1:
        pca = PCA(n_components=2)
        components = pca.fit_transform(final_embeddings_df.values)

        plt.figure(figsize=(10, 8))
        plt.scatter(components[:, 0], components[:, 1], c="blue", alpha=0.7)

        # Print Legend for the User
        print("\n=== LEGEND: Index to Series Name ===")

        # Enumerate gives us (0, "series_A"), (1, "series_B")
        for i, series_name in enumerate(final_embeddings_df.index):
            # Print mapping to console
            if i < 20:  # Only print first 20 to avoid spamming console
                print(f"Index {i}: {series_name}")
            elif i == 20:
                print("... (see dataframe for full list)")

            # Annotate plot with INDEX number
            plt.annotate(
                str(i),
                (components[i, 0], components[i, 1]),
                fontsize=9,
                fontweight="bold",
            )

        plt.title(
            f"Time Series Embeddings (Labeled by Index)\nLen: {CONFIG['seq_len']}, Dim: {CONFIG['embedding_dim']}"
        )
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.grid(True, alpha=0.3)

        plt.savefig(CONFIG["output_plot"])
        print(f"\nPlot saved to {CONFIG['output_plot']}")
        plt.show()
    else:
        print("Not enough series to perform PCA.")

    print("\nDone!")

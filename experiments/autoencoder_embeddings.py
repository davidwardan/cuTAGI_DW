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
    "output_plot": "out/embeddings_pca.svg",
    "output_embedding_file": "out/autoencoder_embeddings.npy",
    # Model Params
    "seq_len": 52,  # Window size
    "embedding_dim": 15,  # Latent vector size
    "hidden_dim": 40,  # Internal LSTM size
    # Training Params
    "batch_size": 128,
    "learning_rate": 1e-3,
    "epochs": 100,
    "patience": 10,
    "teacher_forcing_start": 0.5,  # Start using Ground Truth 50% of the time
    "generate_dummy_data": False,  # Set True if you don't have files yet
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
                # Sine waves
                ts = np.sin(np.linspace(0, 20, length))
            else:
                # Linear trends with noise
                ts = np.linspace(0, 1, length) + np.random.normal(0, 0.1, length)
            # Pad with NaNs for realism
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

    # SAFETY: Only process numeric columns
    numeric_df = df.select_dtypes(include=[np.number])

    for col in numeric_df.columns:
        # 1. Remove trailing NaNs
        series_values = numeric_df[col].to_numpy(dtype=np.float32, copy=True)
        raw_values = _trim_trailing_nans(series_values)

        if raw_values.size < seq_len:
            continue

        finite_values = raw_values[np.isfinite(raw_values)]
        if finite_values.size < seq_len:
            continue

        # 2. Standardization (Z-Score)
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
# 3. PYTORCH MODEL (RECURRENT DECODER)
# ==========================================


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
        self.encoder_fc = nn.Linear(hidden_dim, embedding_dim)

        # --- DECODER SETUP ---
        # Map embedding back to hidden/cell state size for initialization
        self.decoder_fc_hidden = nn.Linear(embedding_dim, hidden_dim)
        self.decoder_fc_cell = nn.Linear(embedding_dim, hidden_dim)

        # LSTM Cell for step-by-step decoding
        self.decoder_cell = nn.LSTMCell(input_size=n_features, hidden_size=hidden_dim)

        # Project hidden state to output value
        self.output_layer = nn.Linear(hidden_dim, n_features)

    def forward(self, x, teacher_forcing_ratio=0.0):
        """
        x: (Batch, Seq_Len, Features)
        teacher_forcing_ratio: Probability of using ground truth input.
        """
        batch_size = x.shape[0]

        # 1. ENCODE
        # output: (batch, seq, hidden), (h_n, c_n)
        _, (hidden, _) = self.encoder_lstm(x)

        # Take the last layer's hidden state
        hidden = hidden.squeeze(0)  # (batch, hidden)

        # Create Embedding
        embedding = self.encoder_fc(hidden)  # (batch, embedding_dim)

        # 2. DECODE INITIALIZATION
        # Initialize Decoder Memory from Embedding
        dec_hidden = self.decoder_fc_hidden(embedding)
        dec_cell = self.decoder_fc_cell(embedding)

        # Start Token (Zero vector)
        current_input = torch.zeros((batch_size, self.n_features), device=x.device)

        outputs = []

        # 3. DECODE LOOP
        for t in range(self.seq_len):
            # Recurrent Step
            dec_hidden, dec_cell = self.decoder_cell(
                current_input, (dec_hidden, dec_cell)
            )

            # Prediction
            reconstruction_t = self.output_layer(dec_hidden)
            outputs.append(reconstruction_t.unsqueeze(1))

            # Determine next input (Teacher Forcing vs Autoregression)
            if self.training and np.random.random() < teacher_forcing_ratio:
                # Use actual ground truth from step t
                current_input = x[:, t, :]
            else:
                # Use own prediction
                current_input = reconstruction_t

        # Combine steps back into sequence
        reconstructed_seq = torch.cat(outputs, dim=1)

        return reconstructed_seq, embedding


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
        # Fallback if file not found
        print(f"File {CONFIG['train_csv']} not found. Generating dummy data...")
        create_dummy_csv(CONFIG["train_csv"], CONFIG["val_csv"])

    df_train = pd.read_csv(CONFIG["train_csv"])
    X_train_np, train_names = prepare_data(df_train, CONFIG["seq_len"])

    # B. Load Validation Data
    if not os.path.exists(CONFIG["val_csv"]):
        create_dummy_csv(CONFIG["train_csv"], CONFIG["val_csv"])

    df_val = pd.read_csv(CONFIG["val_csv"])
    X_val_np, _ = prepare_data(df_val, CONFIG["seq_len"])

    print(f"Train Windows: {len(X_train_np)} | Val Windows: {len(X_val_np)}")

    if len(X_train_np) == 0:
        raise ValueError("No training windows generated. Check data format.")

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

    # --- Step 3: Training Loop ---
    print("\nStarting Training with Recurrent Decoder...")

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    for epoch in range(CONFIG["epochs"]):
        # A. Calculate Teacher Forcing Ratio (Linear Decay)
        # Starts at CONFIG['teacher_forcing_start'] and goes to 0 by the end
        tf_ratio = max(
            0.0, CONFIG["teacher_forcing_start"] * (1 - epoch / CONFIG["epochs"])
        )

        # B. TRAINING PHASE
        model.train()
        train_loss = 0
        for batch_in, batch_target in train_loader:
            optimizer.zero_grad()

            # Pass teacher_forcing_ratio to forward
            reconstruction, _ = model(batch_in, teacher_forcing_ratio=tf_ratio)

            loss = criterion(reconstruction, batch_target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # C. VALIDATION PHASE (No Teacher Forcing)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_in, batch_target in val_loader:
                # Force ratio 0.0 for validation
                reconstruction, _ = model(batch_in, teacher_forcing_ratio=0.0)
                loss = criterion(reconstruction, batch_target)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(
            f"Epoch [{epoch+1}/{CONFIG['epochs']}] "
            f"TF: {tf_ratio:.2f} | "
            f"Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}"
        )

        # D. EARLY STOPPING CHECK
        early_stop_delta = 1e-5
        if (avg_val_loss - best_val_loss) < early_stop_delta:
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
            _, batch_embeddings = model(batch_in, teacher_forcing_ratio=0.0)
            embedding_list.append(batch_embeddings.cpu().numpy())

    all_embeddings = np.concatenate(embedding_list, axis=0)

    # --- Step 5: Aggregation ---
    res_df = pd.DataFrame(all_embeddings)
    res_df["series_name"] = train_names
    final_embeddings_df = res_df.groupby("series_name").mean()

    # --- Step 6: PCA Visualization ---
    if len(final_embeddings_df) > 1:
        pca = PCA(n_components=2)
        components = pca.fit_transform(final_embeddings_df.values)

        plt.figure(figsize=(10, 8))
        plt.scatter(components[:, 0], components[:, 1], c="blue", alpha=0.7)

        # Print Legend (Subset)
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
            f"Embeddings (Recurrent Decoder)\nLen: {CONFIG['seq_len']}, Dim: {CONFIG['embedding_dim']}"
        )
        plt.xlabel("PC 1")
        plt.ylabel("PC 2")
        plt.grid(True, alpha=0.3)

        plt.savefig(CONFIG["output_plot"], bbox_inches="tight")
        plt.close()
        print(f"\nPlot saved to {CONFIG['output_plot']}")
        # plt.show() # Uncomment if running in interactive mode
    else:
        print("Not enough series to perform PCA.")

    # --- Step 7: Save Embeddings ---
    print(f"Saving embeddings to {CONFIG['output_embedding_file']}...")
    np.save(CONFIG["output_embedding_file"], final_embeddings_df.values)
    print("Embeddings saved.")

    print("\nDone!")

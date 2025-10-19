import numpy as np
import pandas as pd
import os
from typing import Dict, Tuple, Optional


class EmbeddingLayer:
    """
    Class to handle a single embedding matrix (mu and var) for one category.
    This is a refactor of your original TimeSeriesEmbeddings class.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_size: int,
        encoding_type: str = None,
        seed: int = None,
    ):
        """
        Initializes the embedding matrix.

        Args:
            num_embeddings (int): The number of unique items to embed (e.g., max(dam_id) + 1).
            embedding_size (int): The dimensionality of the embedding vector.
            encoding_type (str, optional): Initialization strategy.
            seed (int, optional): Random seed.
        """
        if seed is not None:
            np.random.seed(seed)

        self.embedding_dim = (num_embeddings, embedding_size)

        if encoding_type == "normal":
            self.mu = np.random.randn(*self.embedding_dim)
            self.var = np.full(self.embedding_dim, 1.0)
        elif encoding_type == "onehot":
            epsilon = 1e-6
            if num_embeddings != embedding_size:
                raise ValueError(
                    "For 'onehot' encoding, num_embeddings must equal embedding_size."
                )
            self.mu = np.full(self.embedding_dim, epsilon)
            np.fill_diagonal(self.mu, 1.0)
            self.var = np.ones(self.embedding_dim)
        elif encoding_type == "uniform":
            self.mu = np.full(self.embedding_dim, 1 / self.embedding_dim[1])
            self.var = np.ones(self.embedding_dim)
        elif encoding_type == "sphere":
            radius = 3
            vecs = np.random.randn(*self.embedding_dim)
            vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
            self.mu = radius * vecs
            self.var = np.full(self.embedding_dim, 1.0)
        else:
            self.mu = np.full(self.embedding_dim, 1.0, dtype=np.float32)
            self.var = np.ones(self.embedding_dim, dtype=np.float32)

        self.mu = self.mu.astype(np.float32)
        self.var = self.var.astype(np.float32)

    def __call__(self, idx: np.ndarray) -> tuple:
        """
        Fetches the embeddings for the given indices.

        Args:
            idx (np.ndarray): An array of indices to fetch.

        Returns:
            tuple: (mu[idx], var[idx])
        """
        return self.mu[idx], self.var[idx]

    def update(self, idx: int, mu_delta: np.ndarray, var_delta: np.ndarray):
        """
        Applies a delta update to a single embedding vector.
        """
        self.mu[idx] = self.mu[idx] + mu_delta
        self.var[idx] = self.var[idx] + var_delta

    def save(self, out_file: str):
        """Saves embeddings to a .npz file."""
        np.savez(
            out_file,
            mu=self.mu,
            var=self.var,
        )

    def load(self, in_file: str):
        """Loads embeddings from a .npz file."""
        data = np.load(in_file)
        self.mu = data["mu"]
        self.var = data["var"]
        # Update embedding_dim based on loaded data
        self.embedding_dim = self.mu.shape


class MappedTimeSeriesEmbeddings:
    """
    Manages multiple, shared EmbeddingLayers based on a time series map.
    """

    def __init__(
        self,
        map_file_path: str,
        embedding_sizes: Dict[str, int],
        encoding_types: Optional[Dict[str, str]] = None,
        seed: int = None,
    ):
        """
        Initializes the mapped embeddings manager.

        Args:
            map_file_path (str): Path to the 'ts_embedding_map.csv' file.
            embedding_sizes (Dict[str, int]): A dictionary mapping category names
                (e.g., 'dam_id') to their desired embedding size (e.g., 10).
            encoding_types (Optional[Dict[str, str]], optional): A dictionary
                mapping category names to initialization types. Defaults to None.
            seed (int, optional): Random seed.
        """
        self.ts_map = None
        self.embedding_categories = []
        self.ts_map_info = {}
        self.embeddings = {}

        # Load the map and discover categories
        self._load_map(map_file_path)

        if encoding_types is None:
            encoding_types = {}

        # Check if all required embedding sizes are provided
        for category in self.embedding_categories:
            if category not in embedding_sizes:
                raise ValueError(
                    f"Missing embedding_size for category: '{category}'. "
                    f"Please provide it in the embedding_sizes dictionary."
                )

        # Initialize an EmbeddingLayer for each category
        for category in self.embedding_categories:
            num_embeddings = self.ts_map_info[category]["num_embeddings"]
            embedding_size = embedding_sizes[category]
            encoding_type = encoding_types.get(category, None)  # Default to None

            self.embeddings[category] = EmbeddingLayer(
                num_embeddings=num_embeddings,
                embedding_size=embedding_size,
                encoding_type=encoding_type,
                seed=seed,
            )

    def _load_map(self, map_file_path: str):
        """Loads the ts_embedding_map.csv and infers embedding categories."""
        if not os.path.exists(map_file_path):
            raise FileNotFoundError(f"Map file not found at: {map_file_path}")

        map_df = pd.read_csv(map_file_path)

        if "ts_id" not in map_df.columns:
            raise ValueError("Map file must contain a 'ts_id' column.")

        # Set ts_id as index for fast lookups
        self.ts_map = map_df.set_index("ts_id")

        # All other columns are assumed to be embedding categories
        self.embedding_categories = [
            col for col in self.ts_map.columns if col != "ts_id"
        ]

        # Sort to ensure a consistent concatenation order
        self.embedding_categories.sort()

        self.ts_map_info = {}
        for category in self.embedding_categories:
            max_id = self.ts_map[category].max()
            num_embeddings = int(max_id + 1)  # Ensure it's a plain int
            self.ts_map_info[category] = {"num_embeddings": num_embeddings}

        print(f"Loaded map. Found categories: {self.embedding_categories}")
        print(f"Map info: {self.ts_map_info}")

    def __call__(self, ts_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Constructs the combined embeddings for a batch of time series IDs.

        Args:
            ts_ids (np.ndarray): A 1D array of time series IDs.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - mu_combined: (batch_size, total_embedding_dim)
                - var_combined: (batch_size, total_embedding_dim)
        """
        ts_ids = np.atleast_1d(ts_ids)

        # Get the categorical indices for all ts_ids in the batch
        try:
            batch_map = self.ts_map.loc[ts_ids]
        except KeyError as e:
            print(f"Error: One or more ts_ids not found in the map: {e}")
            raise

        mus = []
        vars_ = []

        # Fetch and collect embeddings for each category
        for category in self.embedding_categories:
            cat_indices = batch_map[category].values
            cat_mu, cat_var = self.embeddings[category](cat_indices)
            mus.append(cat_mu)
            vars_.append(cat_var)

        # Concatenate along the feature dimension
        mu_combined = np.concatenate(mus, axis=1)
        var_combined = np.concatenate(vars_, axis=1)

        return mu_combined, var_combined

    def update(
        self,
        ts_ids: np.ndarray,
        mu_deltas_combined: np.ndarray,
        var_deltas_combined: np.ndarray,
    ):
        """
        Applies batch updates, summing deltas for shared embeddings.

        Args:
            ts_ids (np.ndarray): 1D array of ts_ids in the batch.
            mu_deltas_combined (np.ndarray): Deltas for the combined mu vectors.
                Shape: (batch_size, total_embedding_dim)
            var_deltas_combined (np.ndarray): Deltas for the combined var vectors.
                Shape: (batch_size, total_embedding_dim)
        """
        ts_ids = np.atleast_1d(ts_ids)
        batch_map = self.ts_map.loc[ts_ids]

        current_offset = 0
        for category in self.embedding_categories:
            embedding_layer = self.embeddings[category]
            embedding_size = embedding_layer.embedding_dim[1]

            # 1. Slice the deltas for the current category
            mu_delta_cat = mu_deltas_combined[
                :, current_offset : current_offset + embedding_size
            ]
            var_delta_cat = var_deltas_combined[
                :, current_offset : current_offset + embedding_size
            ]
            current_offset += embedding_size

            # 2. Get the corresponding embedding indices for the batch
            cat_indices = batch_map[category].values

            # 3. Use pandas to group deltas by index and sum them
            # This efficiently handles the "sum updates for the same embedding"
            df_mu_deltas = pd.DataFrame(mu_delta_cat, index=cat_indices)
            df_var_deltas = pd.DataFrame(var_delta_cat, index=cat_indices)

            summed_mu_deltas = df_mu_deltas.groupby(df_mu_deltas.index).sum()
            summed_var_deltas = df_var_deltas.groupby(df_var_deltas.index).sum()

            # 4. Apply the summed updates
            for idx, mu_delta_row in summed_mu_deltas.iterrows():
                mu_d = mu_delta_row.values
                var_d = summed_var_deltas.loc[idx].values
                embedding_layer.update(idx, mu_d, var_d)

    def save(self, out_dir_prefix: str):
        """
        Saves all managed embeddings.

        Args:
            out_dir_prefix (str): A prefix for the output files.
                (e.g., 'my_model/embeddings')
        """
        print(f"Saving embeddings with prefix: {out_dir_prefix}")
        for category, embedding_layer in self.embeddings.items():
            out_file = f"{out_dir_prefix}_{category}.npz"
            embedding_layer.save(out_file)
            print(f"Saved {category} embeddings to {out_file}")

    def load(self, in_dir_prefix: str):
        """
        Loads all managed embeddings. Assumes class was initialized with
        the *same* map and embedding_sizes.

        Args:
            in_dir_prefix (str): The prefix used during saving.
        """
        print(f"Loading embeddings from prefix: {in_dir_prefix}")
        for category, embedding_layer in self.embeddings.items():
            in_file = f"{in_dir_prefix}_{category}.npz"
            if not os.path.exists(in_file):
                print(f"Warning: Could not find embedding file: {in_file}")
                continue

            embedding_layer.load(in_file)
            print(f"Loaded {category} embeddings from {in_file}")

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

    def update(
        self,
        idx: np.array,
        mu_delta: np.ndarray,
        var_delta: np.ndarray,
        lr: float = 1.0,
        skip_update: bool = False,
    ):
        """
        Applies a delta update to embedding vectors.
        Handles batches and sums updates for repeated indices using np.add.at.
        """
        if skip_update:
            return

        # Ensure idx is an array for consistent processing
        idx = np.atleast_1d(idx)

        # Ensure deltas are at least 2D if they are 1D and idx is scalar
        if idx.ndim == 1 and idx.size == 1 and mu_delta.ndim == 1:
            mu_delta = mu_delta.reshape(1, -1)
        if idx.ndim == 1 and idx.size == 1 and var_delta.ndim == 1:
            var_delta = var_delta.reshape(1, -1)

        # Check dimensions match
        if idx.shape[0] != mu_delta.shape[0] or idx.shape[0] != var_delta.shape[0]:
            raise ValueError(
                f"Shape mismatch: idx shape {idx.shape} does not match delta shapes "
                f"{mu_delta.shape} or {var_delta.shape}"
            )

        # Check for negative indexes
        active_mask = idx >= 0

        # Filter indices and deltas
        idx_filtered = idx[active_mask]

        # Check if any valid indices remain
        if idx_filtered.size == 0:
            return

        mu_delta_filtered = mu_delta[active_mask] * lr
        var_delta_filtered = var_delta[active_mask] * lr

        # Use np.add.at to correctly sum updates for repeated indices
        np.add.at(self.mu, idx_filtered, mu_delta_filtered)
        np.add.at(self.var, idx_filtered, var_delta_filtered)

        np.maximum(
            self.var, 1e-5, out=self.var
        )  # prevent variances from going negative

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

    def as_coordinates(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns PCA-reduced embedding coordinates for the requested number of
        principal components.

        Args:
            n (int): Number of principal components (coordinates) to return.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Projected (mu, var) with shape
            (num_embeddings, n).
        """
        _, embedding_size = self.embedding_dim
        if not 0 < n <= embedding_size:
            raise ValueError(
                "Requested coordinate dimensionality must be positive and <= "
                f"embedding_size ({embedding_size})."
            )

        # Center mu before computing PCA
        mu_centered = self.mu - self.mu.mean(axis=0, keepdims=True)

        # Compute principal axes via SVD
        _, _, vt = np.linalg.svd(mu_centered, full_matrices=False)
        components = vt[:n]  # shape: (n, embedding_size)

        # Project means
        mu_coords = mu_centered @ components.T  # (num_embeddings, n)

        # Project variances assuming diagonal covariance per dimension
        comp_sq = components**2
        var_coords = self.var @ comp_sq.T  # (num_embeddings, n)

        return mu_coords.astype(np.float32), var_coords.astype(np.float32)


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

        # Check category order consistency
        print(f"Using category order from CSV: {self.embedding_categories}")

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

        # Ensure ts_id is integer before setting as index
        try:
            map_df["ts_id"] = map_df["ts_id"].astype(int)
        except ValueError:
            print(
                f"Warning: Could not cast 'ts_id' column in {map_file_path} to integer. This may cause lookup errors."
            )

        # Set ts_id as index for fast lookups
        self.ts_map = map_df.set_index("ts_id")

        # All other columns are assumed to be embedding categories
        self.embedding_categories = [
            col for col in self.ts_map.columns if col != "ts_id"
        ]

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
        Concatenation order is determined by self.embedding_categories.

        Args:
            ts_ids (np.ndarray): A 1D array of time series IDs.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - mu_combined: (batch_size, total_embedding_dim)
                - var_combined: (batch_size, total_embedding_dim)
        """
        ts_ids = np.atleast_1d(ts_ids)

        # Ensure ts_ids are int for lookup
        ts_ids_int = ts_ids.astype(int)

        # Get the categorical indices for all ts_ids in the batch
        try:
            batch_map = self.ts_map.loc[ts_ids_int]
        except KeyError as e:
            print(f"Error: One or more ts_ids not found in the map: {e}")
            print(f"Attempted to find: {ts_ids_int}")
            print(f"Available map index: {self.ts_map.index.values}")
            raise

        mus = []
        vars_ = []

        # Fetch and collect embeddings for each category
        # The order of this loop determines the concatenation order
        for category in self.embedding_categories:
            cat_indices = batch_map[category].values
            cat_mu, cat_var = self.embeddings[category](cat_indices)
            mus.append(cat_mu)
            vars_.append(cat_var)

        # Concatenate along the feature dimension
        mu_combined = np.concatenate(mus, axis=1)
        var_combined = np.concatenate(vars_, axis=1)

        return mu_combined, var_combined

    def set_category_order(self, category_order: list):
        """
        Updates the concatenation order of the embedding categories.

        The new order MUST contain the exact same categories as the original map.

        Args:
            category_order (list): A list of category name strings in the
                                   desired new order.
        """
        print(f"Attempting to set new category order: {category_order}")

        # Validate that the new order contains the same categories
        original_categories = set(self.embeddings.keys())
        new_categories = set(category_order)

        if new_categories != original_categories:
            raise ValueError(
                f"New category order {new_categories} does not match "
                f"original categories {original_categories}"
            )

        # All checks passed, update the order
        self.embedding_categories = category_order
        print(f"Successfully set category order to: {self.embedding_categories}")

    def update_map(self, new_map_file_path: str):
        """
        Updates the internal ts_id-to-category mapping from a new CSV file.

        This is intended for scenarios (like testing) where the ts_id-to-category
        relationship might change, but the underlying categorical embeddings
        (e.g., 'wave', 'amplitude') are the same and have already been trained.

        The new map MUST be compatible:
        1. It must contain the exact same category columns as the original map.
        2. The maximum ID for each category in the new map must be LESS THAN
           the 'num_embeddings' the class was initialized with (i.e., no
           out-of-bounds indices).

        Args:
            new_map_file_path (str): Path to the new mapping CSV file.
        """
        print(f"Attempting to update map from: {new_map_file_path}")
        if not os.path.exists(new_map_file_path):
            raise FileNotFoundError(f"New map file not found at: {new_map_file_path}")

        map_df = pd.read_csv(new_map_file_path)

        # Validation Check 1: 'ts_id' column
        if "ts_id" not in map_df.columns:
            raise ValueError("New map file must contain a 'ts_id' column.")

        # Validation Check 2: Category columns
        # TODO: A more robust check might be against self.embeddings.keys()
        new_categories = [col for col in map_df.columns if col != "ts_id"]
        if set(new_categories) != set(self.embedding_categories):
            raise ValueError(
                f"New map categories {set(new_categories)} do not match "
                f"original categories {set(self.embedding_categories)}"
            )

        # Validation Check 3: Max IDs (Index Bounds)
        for category in self.embedding_categories:
            new_max_id = map_df[category].max()
            original_num_embeddings = self.ts_map_info[category]["num_embeddings"]

            # new_max_id must be < original_num_embeddings (since num_embeddings = max_id + 1)
            if new_max_id >= original_num_embeddings:
                raise ValueError(
                    f"Category '{category}': new max_id ({new_max_id}) is "
                    f"out of bounds for original num_embeddings ({original_num_embeddings})."
                )

        # All checks passed, update the map
        try:
            map_df["ts_id"] = map_df["ts_id"].astype(int)
        except ValueError:
            print(
                f"Warning: Could not cast 'ts_id' column in new map to integer. "
                "This may cause lookup errors."
            )

        self.ts_map = map_df.set_index("ts_id")
        print(f"Successfully updated ts_map from {new_map_file_path}")

    def update(
        self,
        ts_ids: np.ndarray,
        mu_deltas_combined: np.ndarray,
        var_deltas_combined: np.ndarray,
    ):
        """
        Applies batch updates, delegating summation for shared embeddings
        to the EmbeddingLayer's update method (which uses np.add.at).
        This method respects the order set in self.embedding_categories.

        Args:
            ts_ids (np.ndarray): 1D array of ts_ids in the batch.
                Shape: (batch_size,)
            mu_deltas_combined (np.ndarray): Deltas for the combined mu vectors.
                Shape: (batch_size, total_embedding_dim)
            var_deltas_combined (np.ndarray): Deltas for the combined var vectors.
                Shape: (batch_size, total_embedding_dim)
        """
        ts_ids = np.atleast_1d(ts_ids)

        # Ensure ts_ids are int for lookup
        ts_ids_int = ts_ids.astype(int)

        # Get the categorical indices for all ts_ids in the batch
        try:
            batch_map = self.ts_map.loc[ts_ids_int]
        except KeyError as e:
            print(f"Error: One or more ts_ids not found in the map: {e}")
            print(f"Attempted to find: {ts_ids_int}")
            print(f"Available map index: {self.ts_map.index.values}")
            raise
        except pd.errors.InvalidIndexError as e:
            # This can happen if ts_ids has duplicates and is not sorted
            # .loc is faster but can be picky. Fallback to reindex.
            print(f"Warning: ts_ids index error ({e}). Falling back to reindex.")
            batch_map = self.ts_map.reindex(ts_ids_int)

        current_offset = 0
        # The order of this loop is critical and matches __call__
        for category in self.embedding_categories:
            embedding_layer = self.embeddings[category]
            embedding_size = embedding_layer.embedding_dim[1]

            # Slice the deltas for the current category
            # Shape: (batch_size, embedding_size)
            mu_delta_cat = mu_deltas_combined[
                :, current_offset : current_offset + embedding_size
            ]
            var_delta_cat = var_deltas_combined[
                :, current_offset : current_offset + embedding_size
            ]
            current_offset += embedding_size

            # Get the corresponding embedding indices for the batch
            # Shape: (batch_size,)
            cat_indices = batch_map[category].values

            # Apply the updates directly
            embedding_layer.update(cat_indices, mu_delta_cat, var_delta_cat)

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

import numpy as np


class TimeSeriesEmbeddings:
    """
    Class to handle embedding operations with mean and variance.
    """

    def __init__(
        self, embedding_dim: tuple, encoding_type: str = None, seed: int = None
    ):

        if seed is not None:
            np.random.seed(seed)
        self.embedding_dim = embedding_dim
        if encoding_type == "normal":
            self.mu_embedding = np.random.randn(*embedding_dim)
            self.var_embedding = np.full(embedding_dim, 1.0)
        elif encoding_type == "onehot":
            epsilon = 1e-6
            if embedding_dim[0] != embedding_dim[1]:
                raise ValueError("Dimensions of the embedding must be equal.")
            self.mu_embedding = np.full(embedding_dim, epsilon)
            np.fill_diagonal(self.mu_embedding, 1.0)
            self.var_embedding = np.ones(embedding_dim)
        elif encoding_type == "uniform":
            self.mu_embedding = np.full(embedding_dim, 1 / (embedding_dim[1]))
            self.var_embedding = np.ones(embedding_dim)
        elif encoding_type == "sphere":
            radius = 3  # radius of the sphere
            # sample means from a normal distribution
            vecs = np.random.randn(*embedding_dim)
            # project each row to the unit sphere
            vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
            # scale to radius R
            self.mu_embedding = radius * vecs
            # small, fixed variance
            self.var_embedding = np.full(embedding_dim, 1.0)
        else:
            self.mu_embedding = np.full(embedding_dim, 1.0, dtype=np.float32)
            self.var_embedding = np.ones(embedding_dim, dtype=np.float32)

        # ensure correct data types
        self.mu_embedding = self.mu_embedding.astype(np.float32)
        self.var_embedding = self.var_embedding.astype(np.float32)

    def __call__(self, idx: int) -> tuple:
        return self.mu_embedding[idx], self.var_embedding[idx]

    def update(self, idx: int, mu_delta: np.ndarray, var_delta: np.ndarray):
        self.mu_embedding[idx] = self.mu_embedding[idx] + mu_delta
        self.var_embedding[idx] = self.var_embedding[idx] + var_delta

    def save(self, out_dir: str):
        np.savez(
            out_dir,
            mu_embedding=self.mu_embedding,
            var_embedding=self.var_embedding,
        )

    def load(self, in_dir: str):
        data = np.load(in_dir)
        self.mu_embedding = data["mu_embedding"]
        self.var_embedding = data["var_embedding"]
